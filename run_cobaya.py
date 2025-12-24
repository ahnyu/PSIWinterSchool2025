import os
import numpy as np
import argparse
import multiprocessing as mp
from multiprocessing import Pool

# Ensure friendly behavior of OpenMP and multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method('spawn', force=True)

import pocomc as pc
from cosmoprimo import Cosmology
from scipy.stats import uniform, norm
from cobaya.model import get_model

# Import likelihood modules (new versions that load data from npz files)
from bao import BAOLikelihood
from cmb import CMBCompressedLikelihood
from supernova import Union3SNLikelihoodSys
from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood

iteration_counter = 0
print_every = 50  # adjust how frequently you want updates

def build_bounds(model, include_bao, bao_sys, include_sn, sn_likelihood, sn_sys, cobaya_nuisance_names=None, cobaya_info=None):
    """
    Build the list of parameter bounds.
    
    Order:
      - Cosmology:
         * For w0waCDM: [w0, wa, Omega_m, omega_b, h]
         * For LCDM: [Omega_m, omega_b, h] (w0=-1, wa=0 fixed)
      - If BAO is included and bao_sys True: add one BAO nuisance parameter.
      - If SN is included: add one SN nuisance parameter and/or one nuisance parameter for systematics.
    """
    bounds = []
    if model == 'w0waCDM':
        # bounds += [[-3, 1], [-3, 2], [0.01, 0.99], [0.005, 0.1], [0.2, 1]]
        bounds += [[-2, 0], [-2, 2], [0.01, 0.99], [0.01, 0.04], [0.4, 1]]
    elif model == 'LCDM':
        bounds += [[0.01, 0.99], [0.005, 0.1], [0.2, 1]]
    else:
        raise ValueError("Unknown model option. Use 'w0waCDM' or 'LCDM'.")
    
    # add As, ns, tau
    bounds += [[1e-10, 5e-9], [0.9, 1.1], [0.01, 0.08]]  # As, ns, tau        

    # add Cobaya nuisance priors 
    if cobaya_nuisance_names is not None and cobaya_info is not None:
        for name in cobaya_nuisance_names:
            par = cobaya_info[name]
            prior = par.get("prior")
            if prior is None:
                raise ValueError(f"No prior for Cobaya parameter {name}")
            # uniform
            if prior.get("dist", "uniform") == "uniform":
                bounds += [[prior["min"], prior["max"]]]
            # normal
            elif prior.get("dist") == "norm":
                mean = prior["loc"]
                sigma = prior["scale"]
                bounds += [[mean - sigma, mean + sigma]]  # uniform window loc +- scale
            else:
                raise ValueError(f"Unknown prior type for {name}")
            
    if include_bao and bao_sys:
        bounds += [[1, 9]]  # BAO nuisance parameter bound (systematics version)
    if include_sn:
        bounds += [[-15, -5]]  # SN nuisance parameter bound
        if sn_sys:
            bounds +=[[1, 5]]  # SN nuisance parameter bound (systematics version)

    # print("bounds:", bounds, "number of sampled params:", len(bounds))
    return bounds

def prepPrior(bounds, bbn=None, bbnidx=3):
    """
    Create a pocomc Prior object for MCMC sampling.
    
    Each parameter gets a prior distribution:
      - Uniform over the provided bounds by default.
      - Optional Gaussian prior for a specific parameter (e.g., from BBN constraints).
    
    Parameters:
    - bounds: list of [min, max] for each parameter
    - bbn: optional [mean, std] for a Gaussian prior on one parameter
    - bbnidx: index of the parameter to apply the Gaussian prior
    
    Returns:
    - pc.Prior object used by PoCoMC to constrain the parameter space before sampling.
    """
    dists = [
        norm(bbn[0], bbn[1]) if (idx == bbnidx and bbn is not None)
        else uniform(lower, upper - lower)
        for idx, (lower, upper) in enumerate(bounds)
    ]
    return pc.Prior(dists)

def total_log_likelihood(params, model, bao_like=None, bao_sys=False, cmb_like=None, sn_like=None, sn_sys=False, cobaya_nuisance_names=None):
    global iteration_counter
    iteration_counter += 1
    """
    Combined likelihood for the chosen probes.
    """

    # print("params:", params)
    idx = 0
    if model == 'w0waCDM':
        if len(params) < 5:
            return -np.inf
        w0, wa, Omega_m, omega_b, h = params[idx:idx+5]
        idx += 5
    elif model == 'LCDM':
        if len(params) < 3:
            return -np.inf
        Omega_m, omega_b, h = params[idx:idx+3]
        w0, wa = -1.0, 0.0
        idx += 3
    else:
        raise ValueError("Unknown cosmological model.")
    
    # Entries for As, ns, tau
    As, ns, tau = params[idx:idx+3]
    idx += 3
    
    # Enforce physical priors
    if (w0 + wa) >= 0:
        return -np.inf
    if (Omega_m * h**2) <= omega_b+0.0006441915396177796:  #correct prior for compressed
        return -np.inf

    # Create a new Cosmology instance with the sampled parameters.
    cosmo = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m, omega_b=omega_b, h=h, mnu=0.06, nnu=3.044)
    cosmo.set_engine('camb')
    
    total_ll = 0.0

    # Update CMB likelihood model with the new cosmo.
    # if cmb_like is not None:
    #     cmb_like.model = cosmo
    #     ll_cmb = cmb_like.calculate()
    #     total_ll += ll_cmb
    #     if total_ll == -np.inf:
    #         return -np.inf
        
    if cmb_like is not None:
        # print('breakpoint1')

        # Map PoCoMC params to Cobaya parameter dict
        point = {}
        # Cosmology
        point.update({
            'w': w0,
            'wa': wa,
            'omch2': Omega_m*h**2 - omega_b,
            'ombh2': omega_b,
            'H0': h*100,
            # Use reference values for As, ns, tau unless you sample them explicitly
            'As': As,
            'ns': ns,
            'tau': tau,
        })

        # Nuisance parameters
        for name in cobaya_nuisance_names:
            if idx >= len(params):
                return -np.inf
            point[name] = params[idx]
            idx += 1
        # print(point, len(point))

        try:
            lp = cmb_like.logposterior(point)        # lp is a dict
            ll_cmb = lp.logpost                     
            if not np.isfinite(ll_cmb):
                # print("ll_cmb=", ll_cmb,'at point=', point)
                return -np.inf
            total_ll += ll_cmb
        except Exception as e:
            print("Cobaya evaluation failed for point:", point)
            print("Exception:", e)
            return -np.inf
        
    
    # Update BAO likelihood model with the new cosmo.
    if bao_like is not None:
        # print('breakpoint2')
        bao_like.model.cosmo = cosmo
        if bao_sys:
            if idx >= len(params):
                print('BAO issue, idx >= len(params)')
                return -np.inf
            bao_nuis = params[idx]
            idx += 1
            ll_bao = bao_like.calculate(sys_coeff=bao_nuis)
        else:
            ll_bao = bao_like.calculate()
        total_ll += ll_bao
        if total_ll == -np.inf:
            print('BAO issue, bao_like=-inf')
            return -np.inf

    
    # SN likelihood: create an instance using the shared cosmo and calculate.
    # if sn_like is not None:
    #     if idx >= len(params):
    #         return -np.inf
    #     sn_nuis = params[idx]
    #     idx += 1
    #     sn_instance = sn_like(cosmo=cosmo)
    #     sn_instance.calculate(dM=sn_nuis)
    #     ll_sn = sn_instance.loglikelihood
    #     total_ll += ll_sn
    #     if total_ll == -np.inf:
    #         return -np.inf

    if sn_like is not None:
        if idx >= len(params):
            return -np.inf
        sn_nuis = params[idx]
        idx += 1
        sn_instance = sn_like(cosmo=cosmo)
        if sn_sys:
            sn_sys_nuis = params[idx]
            idx += 1
            sn_instance.calculate(dM=sn_nuis, sys_coeff=sn_sys_nuis)
        else:
            sn_instance.calculate(dM=sn_nuis)

        ll_sn = sn_instance.loglikelihood
        total_ll += ll_sn
        if total_ll == -np.inf:
            return -np.inf
    
    # print("idx:", idx, "params length:", len(params))
    # print("Cosmo:", w0, wa, Omega_m, omega_b, h, As, ns, tau)

    # Live print progress
    if iteration_counter % print_every == 0:
        print(f"\rIteration {iteration_counter}: total_ll = {total_ll:.2f}, ll_cmb = {ll_cmb:.2f}", end='', flush=True)

    return total_ll

def main(args):
    
    probe_list = [p.strip().upper() for p in args.likelihoods.split(',')]
    include_bao = 'BAO' in probe_list
    include_cmb = 'CMB' in probe_list
    include_sn  = 'SN'  in probe_list
    print('probe list: ',probe_list)
    model = args.model  # 'w0waCDM' or 'LCDM'
    
    # init BAO likelihood.
    bao_like = None
    if include_bao:
        dataset = args.bao_dataset  
        bao_like = BAOLikelihood(data_dir=args.data_dir, dataset=dataset, engine='camb')
    
    # init CMB likelihood.
    cmb_like = None
    # if include_cmb:
    #     cmb_like = CMBCompressedLikelihood(engine='camb', cosmo=None)
    # Define your model and data
    info = {
        "theory": {"camb": None},
        "params": {
            # cosmological parameters
            "w": {"prior": {"min": -2, "max": 0}, "ref": -1},
            "wa": {"prior": {"min": -2, "max": 2}, "ref": 0},
            "omch2": {"prior": {"min": 0.05, "max": 0.25}, "ref": 0.1203},
            "ombh2": {"prior": {"min": 0.01, "max": 0.04}, "ref": 0.02218},
            "H0": {"prior": {"min": 40, "max": 100}, "ref": 67.36}, # check if its zeta instead of H0
            "As": {"prior": {"min": 1e-10, "max": 5e-9}, "ref": 2.1e-9, "latex": "A_s"},
            "ns": {"prior": {"min": 0.9, "max": 1.1}, "ref": 0.965},
            "tau": {"prior": {"min": 0.01, "max": 0.08}, "ref": 0.054},

            # nuisance parameters (Planck 2018 high-l TTTEEE) I used cobaya-doc planck_2018_highl_plik.TTTEEE --python
            "A_cib_217": {"prior": {"dist": "uniform", "min": 0, "max": 200}, "ref": 67, "latex": "A^\\mathrm{CIB}_{217}"},
            "A_planck": {"prior": {"dist": "norm", "loc": 1, "scale": 0.0025}, "ref": 1, "latex": "y_\\mathrm{cal}"},
            "A_sz": {"prior": {"dist": "uniform", "min": 0, "max": 10}, "ref": 7, "latex": "A^\\mathrm{tSZ}_{143}"},
            "calib_100T": {"prior": {"dist": "norm", "loc": 1.0002, "scale": 0.0007}, "ref": 1.0002, "latex": "c_{100}"},
            "calib_217T": {"prior": {"dist": "norm", "loc": 0.99805, "scale": 0.00065}, "ref": 0.99805, "latex": "c_{217}"},

            "gal545_A_100": {"prior": {"dist": "norm", "loc": 8.6, "scale": 2}, "ref": 8.6, "latex": "A^\\mathrm{dustTT}_{100}"},
            "gal545_A_143": {"prior": {"dist": "norm", "loc": 10.6, "scale": 2}, "ref": 10.6, "latex": "A^\\mathrm{dustTT}_{143}"},
            "gal545_A_143_217": {"prior": {"dist": "norm", "loc": 23.5, "scale": 8.5}, "ref": 23.5, "latex": "A^\\mathrm{dustTT}_{143\\times217}"},
            "gal545_A_217": {"prior": {"dist": "norm", "loc": 91.9, "scale": 20}, "ref": 91.9, "latex": "A^\\mathrm{dustTT}_{217}"},

            # fixed EE dust parameters
            "galf_EE_A_100": {"value": 0.055, "ref": 0.055, "latex": "A^\\mathrm{dustEE}_{100}"},
            "galf_EE_A_100_143": {"value": 0.04, "ref": 0.04, "latex": "A^\\mathrm{dustEE}_{100\\times143}"},
            "galf_EE_A_100_217": {"value": 0.094, "ref": 0.094, "latex": "A^\\mathrm{dustEE}_{100\\times217}"},
            "galf_EE_A_143": {"value": 0.086, "ref": 0.086, "latex": "A^\\mathrm{dustEE}_{143}"},
            "galf_EE_A_143_217": {"value": 0.21, "ref": 0.21, "latex": "A^\\mathrm{dustEE}_{143\\times217}"},
            "galf_EE_A_217": {"value": 0.7, "ref": 0.7, "latex": "A^\\mathrm{dustEE}_{217}"},

            # TE dust parameters
            "galf_TE_A_100": {"prior": {"dist": "norm", "loc": 0.13, "scale": 0.042}, "ref": 0.13, "latex": "A^\\mathrm{dustTE}_{100}"},
            "galf_TE_A_100_143": {"prior": {"dist": "norm", "loc": 0.13, "scale": 0.036}, "ref": 0.13, "latex": "A^\\mathrm{dustTE}_{100\\times143}"},
            "galf_TE_A_100_217": {"prior": {"dist": "norm", "loc": 0.46, "scale": 0.09}, "ref": 0.46, "latex": "A^\\mathrm{dustTE}_{100\\times217}"},
            "galf_TE_A_143": {"prior": {"dist": "norm", "loc": 0.207, "scale": 0.072}, "ref": 0.207, "latex": "A^\\mathrm{dustTE}_{143}"},
            "galf_TE_A_143_217": {"prior": {"dist": "norm", "loc": 0.69, "scale": 0.09}, "ref": 0.69, "latex": "A^\\mathrm{dustTE}_{143\\times217}"},
            "galf_TE_A_217": {"prior": {"dist": "norm", "loc": 1.938, "scale": 0.54}, "ref": 1.938, "latex": "A^\\mathrm{dustTE}_{217}"},

            # kSZ and point sources
            "ksz_norm": {"prior": {"dist": "uniform", "min": 0, "max": 10}, "ref": 0, "latex": "A^\\mathrm{kSZ}"},
            "ps_A_100_100": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 257, "latex": "A^\\mathrm{PS}_{100}"},
            "ps_A_143_143": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 47, "latex": "A^\\mathrm{PS}_{143}"},
            "ps_A_143_217": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 40, "latex": "A^\\mathrm{PS}_{143\\times217}"},
            "ps_A_217_217": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 104, "latex": "A^\\mathrm{PS}_{217}"},
            "xi_sz_cib": {"prior": {"dist": "uniform", "min": 0, "max": 1}, "ref": 0, "latex": "\\xi^{\\mathrm{tSZ}\\times\\mathrm{CIB}}"},

        },
        "likelihood": {
            "planck_2018_highl_plik.TTTEEE": None,
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_lensing.clik": None,
        },
        # sampler can be None if I just want likelihoods
        "sampler": None,
        "output": None  # no chains needed
    }
    # Build the model
    cobaya_model = get_model(info)
    # cobaya_model.loglikes()
    cmb_like = cobaya_model

    # List of all parameters with priors (sampled parameters)
    cobaya_nuis_names = [name for name, par in info["params"].items() if "prior" in par]
    cobaya_cosmo_names = ['w', 'wa', 'omch2', 'ombh2', 'H0', 'As', 'ns', 'tau']
    cobaya_nuisance_names = [name for name in cobaya_nuis_names
                         if name not in cobaya_cosmo_names]
    # print("cobaya_nuis_names=",cobaya_nuis_names)
    cobaya_info = info["params"]

    # init SN likelihood.
    sn_like = None
    if include_sn:
        sn_map = {
            'union3': Union3SNLikelihoodSys if args.sn_sys else Union3SNLikelihood,
            'pantheonplus': PantheonPlusSNLikelihood,
            'desy5': DESY5SNLikelihood
        }
        key = args.sn_likelihood.lower()
        if key not in sn_map:
            raise ValueError("Invalid SN likelihood. Choose from: union3, pantheonplus, desy5.")
        sn_like = sn_map[key]
    
    # Build priors
    bounds = build_bounds(model, include_bao, args.bao_sys, include_sn, args.sn_likelihood, args.sn_sys, cobaya_nuisance_names, cobaya_info)
    if include_cmb:
        prior = prepPrior(bounds)
    else:
        if model =='LCDM':
            prior = prepPrior(bounds,[0.02218,0.00055],bbnidx=1)  
        else:
            prior = prepPrior(bounds,[0.02218,0.00055])        
        
    
    # Total likelihood function that updates the shared cosmology.
    def lnlike(params):
        return total_log_likelihood(params, model, bao_like=bao_like, bao_sys= None,
                                    cmb_like=cmb_like, sn_like=sn_like, sn_sys= None, cobaya_nuisance_names=cobaya_nuisance_names)
    # Correct sequence for testparams (using Cobaya reference values)
    testparams = [-1, 0, 0.1203, 0.02218, 67.36, 2.1e-09, 0.965, 0.054, 67, 1, 7, 1.0002, 0.99805, 8.6, 10.6, 23.5, 91.9, 0.13, 0.13, 0.46, 0.207, 0.69, 1.938, 0, 257, 47, 40, 104, 0]
    lp = cmb_like.logposterior(testparams)        # lp is a dict
    cmb_val = lp.logpost
    # bao_indices = [2,3,4,0,1]
    # bao_val = bao_like.calculate([testparams[i] for i in bao_indices])
    # print("CMB:", cmb_val) #, "BAO:", bao_val)
    # print('test loglike:',lnlike(testparams),'at point=',testparams)
    # Use multiprocessing pool.
    # with Pool(args.ncores) as pool:
    sampler = pc.Sampler(
        prior=prior,
        likelihood=total_log_likelihood,
        vectorize=False,
            # pool=pool,
        output_dir=args.output_dir,
        output_label=args.output_label,
        likelihood_kwargs={'model': model,
                        'bao_like': bao_like,
                        'bao_sys': args.bao_sys,
                        'cmb_like': cmb_like,
                        'sn_like': sn_like,
                        'sn_sys': args.sn_sys,
                        'cobaya_nuisance_names': cobaya_nuisance_names}
        )
        # sampler.run(n_total=8192)
    sampler.run(n_total=1000)
    
    samples, weights, logl, logp = sampler.posterior()
    output_file = os.path.join(args.output_dir, args.output_label + '.txt')
    np.savetxt(output_file, np.column_stack((samples, weights, logl, logp)),
               header='samples weight logl logp')
    print(f"Sampling complete. Results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run joint likelihood sampler for BAO, CMB, and SN.")
    parser.add_argument("--data_dir", type=str, default="bao_data",
                        help="Directory containing data files (npz files for BAO, etc.)")
    parser.add_argument("--likelihoods", type=str, default="BAO,CMB,SN",
                        help="Comma-separated list of probes to include (choose from BAO, CMB, SN)")
    parser.add_argument("--bao_dataset", type=str, default="DESI_SDSS",
                        help="BAO dataset to use: 'DESI_SDSS' or 'DESI'")
    parser.add_argument("--bao_sys", action="store_true",
                        help="Use BAO likelihood with systematics (adds one extra parameter)")
    parser.add_argument("--sn_likelihood", type=str, default="union3",
                        help="SN likelihood to use: union3, pantheonplus, or desy5")
    parser.add_argument("--sn_sys", action="store_true",
                    help="Use SN likelihood with systematics (adds one extra parameter)")
    parser.add_argument("--model", type=str, default="w0waCDM",
                        help="Cosmological model: 'w0waCDM' (sample w0,wa) or 'LCDM' (fix w0=-1, wa=0)")
    parser.add_argument("--output_dir", type=str, default="chains/",
                        help="Directory to save the chain files")
    parser.add_argument("--output_label", type=str, default="chain_joint",
                        help="Label for the output chain file")
    parser.add_argument("--ncores", type=int, default=16,
                        help="Number of cores to use for parallel processing")
    args = parser.parse_args()
    main(args)