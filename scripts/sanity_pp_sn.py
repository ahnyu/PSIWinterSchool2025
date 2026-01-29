import os
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
import numpy as np
import argparse
import multiprocessing as mp
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method("spawn", force=True)

import pocomc as pc
from cosmoprimo import Cosmology
from scipy.stats import uniform, norm

from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood
from compressed_sn import CompressedPantheonPlusSNLikelihood


_SN_INSTANCE = None
_SN_CLASS = None

def _get_sn_instance(sn_like_cls, cosmo, **init_kwargs):
    """
    Lazily construct the SN likelihood once per process, then reuse it.
    If (re)constructed, pass init_kwargs (e.g., compressed_fn for compressed SN).
    """
    global _SN_INSTANCE, _SN_CLASS
    if sn_like_cls is None:
        return None

    if _SN_INSTANCE is None or (_SN_CLASS is not sn_like_cls):
        _SN_INSTANCE = sn_like_cls(cosmo=cosmo, **init_kwargs)
        _SN_CLASS = sn_like_cls
        return _SN_INSTANCE

    # Update cosmo in-place (works for desilike SN likelihoods)
    _SN_INSTANCE.cosmo = cosmo
    return _SN_INSTANCE


def build_bounds(model, include_sn=True):
    bounds = []
    if model == "w0waCDM":
        bounds += [[-3, 1], [-3, 2], [0.01, 0.99], [0.005, 0.1], [0.2, 1]]  # w0, wa, Om, ob, h
    elif model == "LCDM":
        bounds += [[0.01, 0.99], [0.005, 0.1], [0.2, 1]]  # Om, ob, h
    else:
        raise ValueError("Unknown model option. Use 'w0waCDM' or 'LCDM'.")

    if include_sn:
        bounds += [[-20, -18]]  # dM / Mb
    return bounds


def prepPrior(bounds, bbn=None, bbnidx=3):
    dists = [
        norm(bbn[0], bbn[1]) if (idx == bbnidx and bbn is not None)
        else uniform(lower, upper - lower)
        for idx, (lower, upper) in enumerate(bounds)
    ]
    return pc.Prior(dists)


def sn_log_likelihood(params, model, sn_like_cls, sn_init_kwargs=None):
    params = np.asarray(params, dtype=float)
    idx = 0

    if model == "w0waCDM":
        if params.size < 6:
            return -np.inf
        w0, wa, Omega_m, omega_b, h = params[idx:idx+5]; idx += 5
    elif model == "LCDM":
        if params.size < 4:
            return -np.inf
        Omega_m, omega_b, h = params[idx:idx+3]; idx += 3
        w0, wa = -1.0, 0.0
    else:
        raise ValueError("Unknown cosmological model.")

    if (w0 + wa) >= 0.0:
        return -np.inf
    omega_nu = 0.0006441915396177796
    if (Omega_m * h**2) <= (omega_b + omega_nu):
        return -np.inf

    dM = float(params[idx])

    try:
        cosmo = Cosmology(
            w0_fld=float(w0), wa_fld=float(wa),
            Omega_m=float(Omega_m), omega_b=float(omega_b), h=float(h),
            mnu=0.06, nnu=3.044
        )
        cosmo.set_engine("camb")

        sn = _get_sn_instance(sn_like_cls, cosmo, **(sn_init_kwargs or {}))

        # keep keyword to avoid signature surprises
        sn.calculate(dM)

        ll = float(sn.loglikelihood)
        if not np.isfinite(ll):
            return -np.inf
        return ll

    except Exception:
        return -np.inf


def main(args):
    sn_map = {
        "pantheonplus": PantheonPlusSNLikelihood,
        "pantheonplus_compressed": CompressedPantheonPlusSNLikelihood,
    }

    key = args.sn_likelihood.lower()
    if key not in sn_map:
        raise ValueError("Invalid --sn_likelihood. Choose from: pantheonplus, pantheonplus_compressed.")
    sn_like_cls = sn_map[key]

    sn_init_kwargs = {}
    if key == "pantheonplus_compressed":
        if args.sn_compressed_fn is None:
            raise ValueError("--sn_compressed_fn is required when --sn_likelihood pantheonplus_compressed")
        sn_init_kwargs["compressed_fn"] = args.sn_compressed_fn

    bounds = build_bounds(args.model, include_sn=True)

    if args.model == "LCDM":
        prior = prepPrior(bounds, bbn=[0.02218, 0.00055], bbnidx=1)  # omega_b index=1
    else:
        prior = prepPrior(bounds, bbn=[0.02218, 0.00055], bbnidx=3)  # omega_b index=3

    os.makedirs(args.output_dir, exist_ok=True)

    with Pool(args.ncores) as pool:
        sampler = pc.Sampler(
            prior=prior,
            likelihood=sn_log_likelihood,
            vectorize=False,
            pool=pool,
            output_dir=args.output_dir,
            output_label=args.output_label,
            likelihood_kwargs={
                "model": args.model,
                "sn_like_cls": sn_like_cls,
                "sn_init_kwargs": sn_init_kwargs,
            },
        )
        sampler.run(n_total=args.n_total)

    samples, weights, logl, logp = sampler.posterior()
    out = os.path.join(args.output_dir, args.output_label + ".txt")
    np.savetxt(out, np.column_stack((samples, weights, logl, logp)),
               header="samples weight logl logp")
    print(f"Done. Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal PantheonPlus SN run for posterior sanity check.")
    parser.add_argument("--sn_likelihood", type=str, default="pantheonplus",
                        help="pantheonplus or pantheonplus_compressed")
    parser.add_argument("--sn_compressed_fn", type=str, default=None,
                        help="Path to .npz for compressed PantheonPlus (required if pantheonplus_compressed)")
    parser.add_argument("--model", type=str, default="w0waCDM",
                        help="w0waCDM or LCDM")
    parser.add_argument("--n_total", type=int, default=2048,
                        help="Total likelihood calls for pocomc")
    parser.add_argument("--ncores", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="_out/")
    parser.add_argument("--output_label", type=str, default="sn_only")
    args = parser.parse_args()
    main(args)
