import os
import numpy as np
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import time


# Ensure friendly behavior of OpenMP and multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method('spawn', force=True)

import pocomc as pc
from cosmoprimo import Cosmology
from scipy.stats import uniform, norm

# Import likelihood modules (new versions that load data from npz files)
from bao import BAOLikelihood
from cmb import CMBCompressedLikelihood
from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood

_SN_INSTANCE = None
_SN_CLASS = None

def _get_sn_instance(sn_like_cls, cosmo):
    """
    Lazily construct the SN likelihood once per process, then reuse it.
    We update the cosmology handle each call (best-effort).
    """
    global _SN_INSTANCE, _SN_CLASS

    if sn_like_cls is None:
        return None

    # (Re)create if first time or class changed
    if _SN_INSTANCE is None or (_SN_CLASS is not sn_like_cls):
        _SN_INSTANCE = sn_like_cls(cosmo=cosmo)
        _SN_CLASS = sn_like_cls
        return _SN_INSTANCE

    # Update cosmology in-place (best-effort)
    _SN_INSTANCE.cosmo = cosmo

    return _SN_INSTANCE


def benchmark_likelihood(ncall, bounds, model, bao_like, bao_sys, cmb_like, sn_like, seed=0):
    """
    Simple benchmark: draw params uniformly in bounds and time ncall evaluations.
    Prints total time, avg time per call, and acceptance rate (finite loglike).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    width = hi - lo

    # small warmup (loads tables / first CAMB calls etc.)
    for _ in range(10):
        p = lo + width * rng.random(size=lo.size)
        _ = total_log_likelihood(p, model, bao_like=bao_like, bao_sys=bao_sys,
                                 cmb_like=cmb_like, sn_like=sn_like)

    t0 = time.perf_counter()
    n_finite = 0
    n_inf = 0
    for _ in range(ncall):
        p = lo + width * rng.random(size=lo.size)
        ll = total_log_likelihood(p, model, bao_like=bao_like, bao_sys=bao_sys,
                                  cmb_like=cmb_like, sn_like=sn_like)
        if np.isfinite(ll):
            n_finite += 1
        else:
            n_inf += 1
    dt = time.perf_counter() - t0

    avg = dt / ncall
    print("\n=== Likelihood benchmark ===")
    print(f"ncall = {ncall}")
    print(f"total time = {dt:.3f} s")
    print(f"avg per call = {avg:.6f} s  ({1.0/avg:.2f} calls/s)")
    print(f"finite = {n_finite}  (-inf/NaN = {n_inf})  accept rate = {n_finite/ncall:.3f}")
    print("=== End benchmark ===\n")


def build_bounds(model, include_bao, bao_sys, include_sn, sn_likelihood):
    """
    Build the list of parameter bounds.
    
    Order:
      - Cosmology:
         * For w0waCDM: [w0, wa, Omega_m, omega_b, h]
         * For LCDM: [Omega_m, omega_b, h] (w0=-1, wa=0 fixed)
      - If BAO is included and bao_sys True: add one BAO nuisance parameter.
      - If SN is included: add one SN nuisance parameter.
    """
    bounds = []
    if model == 'w0waCDM':
        bounds += [[-3, 1], [-3, 2], [0.01, 0.99], [0.005, 0.1], [0.2, 1]]
    elif model == 'LCDM':
        bounds += [[0.01, 0.99], [0.005, 0.1], [0.2, 1]]
    else:
        raise ValueError("Unknown model option. Use 'w0waCDM' or 'LCDM'.")
    
    if include_bao and bao_sys:
        bounds += [[1, 9]]  # BAO nuisance parameter bound (systematics version)
    if include_sn:
        bounds += [[-15, -5]]  # SN nuisance parameter bound
    print(bounds)
    return bounds

def prepPrior(bounds, bbn=None, bbnidx=3):
    """
    Build a pocomc Prior object from bounds.
    """
    dists = [
        norm(bbn[0], bbn[1]) if (idx == bbnidx and bbn is not None)
        else uniform(lower, upper - lower)
        for idx, (lower, upper) in enumerate(bounds)
    ]
    return pc.Prior(dists)

def total_log_likelihood(params, model, bao_like=None, bao_sys=False, cmb_like=None, sn_like=None):
    idx = 0
    params = np.asarray(params, dtype=float)

    if model == 'w0waCDM':
        if params.size < 5: return -np.inf
        w0, wa, Omega_m, omega_b, h = params[idx:idx+5]; idx += 5
    elif model == 'LCDM':
        if params.size < 3: return -np.inf
        Omega_m, omega_b, h = params[idx:idx+3]; idx += 3
        w0, wa = -1.0, 0.0
    else:
        raise ValueError("Unknown cosmological model.")

    if (w0 + wa) >= 0.0:
        return -np.inf
    omega_nu = 0.0006441915396177796  # ~ 0.06/93.14
    if (Omega_m * h**2) <= (omega_b + omega_nu):
        return -np.inf

    try:
        cosmo = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m, omega_b=omega_b, h=h, mnu=0.06, nnu=3.044)
        cosmo.set_engine('camb')

        total_ll = 0.0

        if bao_like is not None:
            bao_like.model.cosmo = cosmo
            if bao_sys:
                if idx >= params.size: return -np.inf
                s = params[idx]; idx += 1
                ll_bao = bao_like.calculate(sys_coeff=s)
            else:
                ll_bao = bao_like.calculate()
            if not np.isfinite(ll_bao): return -np.inf
            total_ll += ll_bao

        if cmb_like is not None:
            cmb_like.model = cosmo
            ll_cmb = cmb_like.calculate()
            if not np.isfinite(ll_cmb): return -np.inf
            total_ll += ll_cmb

        if sn_like is not None:
            if idx >= len(params):
                return -np.inf
            sn_nuis = params[idx]
            idx += 1    
            sn_instance = _get_sn_instance(sn_like, cosmo)
            sn_instance.calculate(sn_nuis)
            ll_sn = sn_instance.loglikelihood
            if not np.isfinite(ll_sn):
                return -np.inf
            total_ll += ll_sn

        return float(total_ll)

    except Exception:
        return -np.inf

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
    sn_like = None
    if include_sn:
        sn_map = {
            'union3': Union3SNLikelihood,
            'pantheonplus': PantheonPlusSNLikelihood,
            'desy5': DESY5SNLikelihood
        }
        key = args.sn_likelihood.lower()
        if key not in sn_map:
            raise ValueError("Invalid SN likelihood. Choose from: union3, pantheonplus, desy5.")
        sn_like = sn_map[key]
    
    # Build priors
    bounds = build_bounds(model, include_bao, args.bao_sys, include_sn, args.sn_likelihood)

    if args.benchmark:
        benchmark_likelihood(
            ncall=1000,
            bounds=bounds,
            model=model,
            bao_like=bao_like,
            bao_sys=args.bao_sys,
            cmb_like=cmb_like,
            sn_like=sn_like,
            seed=0,
        )
        return

    
    if include_cmb:
        prior = prepPrior(bounds)
    else:
        if model =='LCDM':
            prior = prepPrior(bounds,[0.02218,0.00055],bbnidx=1)  
        else:
            prior = prepPrior(bounds,[0.02218,0.00055])        
        
    
    # Total likelihood function that updates the shared cosmology.
    def lnlike(params):
        return total_log_likelihood(params, model, bao_like=bao_like, bao_sys=args.bao_sys,
                                    cmb_like=cmb_like, sn_like=sn_like)
    # Use multiprocessing pool.
    with Pool(args.ncores) as pool:
        sampler = pc.Sampler(
            prior=prior,
            likelihood=total_log_likelihood,
            vectorize=False,
            pool=pool,
            output_dir=args.output_dir,
            output_label=args.output_label,
            likelihood_kwargs={'model':model,
                               'bao_like':bao_like,
                               'bao_sys':args.bao_sys,
                               'cmb_like':cmb_like,
                               'sn_like':sn_like}
        )
        sampler.run(n_total=8192)
    
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
    parser.add_argument("--model", type=str, default="w0waCDM",
                        help="Cosmological model: 'w0waCDM' (sample w0,wa) or 'LCDM' (fix w0=-1, wa=0)")
    parser.add_argument("--output_dir", type=str, default="chains/",
                        help="Directory to save the chain files")
    parser.add_argument("--output_label", type=str, default="chain_joint",
                        help="Label for the output chain file")
    parser.add_argument("--ncores", type=int, default=16,
                        help="Number of cores to use for parallel processing")
    parser.add_argument("--benchmark", action="store_true",
                    help="Benchmark: run 1000 likelihood evaluations and exit")
    args = parser.parse_args()
    main(args)
