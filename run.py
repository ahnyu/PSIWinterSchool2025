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

# Import likelihood modules (new versions that load data from npz files)
from bao import BAOLikelihood
from cmb import CMBCompressedLikelihood
from supernova import Union3SNLikelihoodSys
from supernova_combined import SN_Combined
from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood
sn_data = np.load("SN_dataset.npy", allow_pickle=True).item()

sn_likelihoods = {}

for name, data in sn_data.items():
    _, z, mu, _, cov = data
    sn_likelihoods[name.lower()] = SN_Combined(
        z=z,
        mu=mu,
        cov=cov,
        name=name.lower()
    )
# print(sn_likelihoods)

def build_bounds(model, include_bao, bao_sys, sn_likes, sn_sys):
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
        bounds += [[-3, 1], [-3, 2], [0.01, 0.99], [0.005, 0.1], [0.2, 1]]
        # bounds += [[-3, 1], [-3, 2], [0.1, 0.99], [0.005, 0.1], [0.2, 1]]

    elif model == 'LCDM':
        bounds += [[0.01, 0.99], [0.005, 0.1], [0.2, 1]]
    else:
        raise ValueError("Unknown model option. Use 'w0waCDM' or 'LCDM'.")
    
    if include_bao and bao_sys:
        bounds += [[1, 9]]  # BAO nuisance parameter bound (systematics version)
    
    if sn_likes:
        for name in sn_likes:
            bounds += [[-15, -5]]  # one dM per SN dataset
        if sn_sys:
            bounds +=[[1, 5]]  # SN nuisance parameter bound (systematics version)

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

def total_log_likelihood(params, model, bao_like=None, bao_sys=False, cmb_like=None, sn_like=None, sn_sys=False):
    """
    Combined likelihood for the chosen probes.
    """

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
    
    
    # Enforce physical priors
    if (w0 + wa) >= 0:
        return -np.inf
    if (Omega_m * h**2) <= omega_b+0.0006441915396177796: 
        return -np.inf
    

    # # Create a new Cosmology instance with the sampled parameters.
    # cosmo = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m, omega_b=omega_b, h=h, mnu=0.06, nnu=3.044)
    # cosmo.set_engine('camb')

    try:
        cosmo = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m,
                          omega_b=omega_b, h=h, mnu=0.06, nnu=3.044, tau_reio=0.0544)
        cosmo.set_engine('camb')
    # except Exception:
    #     return -np.inf  # catch invalid cosmology
    except Exception as e:
        if 'reionization' in str(e).lower() or 'optical depth' in str(e).lower():
            return -np.inf
        raise
    
    total_ll = 0.0

    # Update CMB likelihood model with the new cosmo.
    if cmb_like is not None:
        cmb_like.model = cosmo
        ll_cmb = cmb_like.calculate()
        total_ll += ll_cmb
        if total_ll == -np.inf:
            return -np.inf
    
    # Update BAO likelihood model with the new cosmo.
    if bao_like is not None:
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

    # if sn_like is not None:
    #     if idx >= len(params):
    #         return -np.inf
    #     sn_nuis = params[idx]
    #     idx += 1
    #     sn_instance = sn_like(cosmo=cosmo)
    #     if sn_sys:
    #         sn_sys_nuis = params[idx]
    #         idx += 1
    #         sn_instance.calculate(dM=sn_nuis, sys_coeff=sn_sys_nuis)
    #     else:
    #         sn_instance.calculate(dM=sn_nuis)

    #     ll_sn = sn_instance.loglikelihood
    #     total_ll += ll_sn
    #     if total_ll == -np.inf:
    #         return -np.inf
        
    # if sn_like is not None:
    #     if idx >= len(params):
    #         return -np.inf

    #     dM = params[idx]
    #     idx += 1

    #     if sn_sys:
    #         if idx >= len(params):
    #             return -np.inf
    #         sys_coeff = params[idx]
    #         idx += 1
    #     else:
    #         sys_coeff = None

    #     ll_sn = sn_like.calculate(cosmo, dM=dM, sys_coeff=sys_coeff)
    #     total_ll += ll_sn

    #     if not np.isfinite(total_ll):
    #         return -np.inf
        
    if sn_like is not None:
        for name, sn in sn_like.items():
            if idx >= len(params):
                return -np.inf

            dM = params[idx]
            idx += 1

            ll_sn = sn.calculate(cosmo, dM=dM)
            if not np.isfinite(ll_sn):
                print(f"SN failure in {name}, dM={dM}")
                return -np.inf

            total_ll += ll_sn

            if not np.isfinite(total_ll):
                return -np.inf 
    
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
    if include_cmb:
        cmb_like = CMBCompressedLikelihood(engine='camb', cosmo=None)

    
    # init SN likelihood.
    # sn_like = None
    # if include_sn:
    #     sn_map = {
    #         'union3': Union3SNLikelihoodSys if args.sn_sys else Union3SNLikelihood,
    #         'pantheonplus': PantheonPlusSNLikelihood,
    #         'desy5': DESY5SNLikelihood
    #     }
    #     key = args.sn_likelihood.lower()
    #     if key not in sn_map:
    #         raise ValueError("Invalid SN likelihood. Choose from: union3, pantheonplus, desy5.")
    #     sn_like = sn_map[key]

    # sn_like = None
    # if include_sn:
    #     key = args.sn_likelihood.lower()
    #     if key not in sn_likelihoods:
    #         raise ValueError(f"Unknown SN dataset: {key}")
    #     sn_like = sn_likelihoods[key]

    sn_likes = {}
    if include_sn:
        requested = [k.strip().lower() for k in args.sn_likelihood.split(",")]

        for key in requested:
            if key not in sn_likelihoods:
                raise ValueError(f"Unknown SN dataset: {key}")
            sn_likes[key] = sn_likelihoods[key]

    
    # Build priors
    bounds = build_bounds(model, include_bao, args.bao_sys, sn_likes, args.sn_sys)
    if include_cmb:
        prior = prepPrior(bounds)
    else:
        if model =='LCDM':
            prior = prepPrior(bounds,[0.02218,0.00055],bbnidx=1)  
        else:
            prior = prepPrior(bounds,[0.02218,0.00055])    

    # # sanity check
    # # --- SANITY CHECK: make sure some prior samples give finite likelihood ---
    # print("Checking prior samples for finite likelihood...")
    # for _ in range(10):
    #     x = prior.rvs()  # draw a random point from the prior
    #     ll = total_log_likelihood(
    #         x,
    #         model,
    #         bao_like=bao_like,
    #         sn_like=sn_like,
    #         sn_sys=args.sn_sys
    #     )
    # print(x, ll)

    print("Number of SN datasets:", len(sn_likes))
    print("Number of SN dM parameters:", len(sn_likes))
    

    # Use multiprocessing pool.
    with Pool(args.ncores) as pool:
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
                            'sn_like': sn_likes,
                            'sn_sys': args.sn_sys}
            )
        # sampler.run(n_total=8192)
        sampler.run(n_total=10)
    
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
    parser.add_argument("--sn_likelihood", type=str, default="union3, pantheonplus, desy5sn",
                        help="SN likelihood to use: union3, pantheonplus, or desy5sn")
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