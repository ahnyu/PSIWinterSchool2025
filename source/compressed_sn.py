import numpy as np

from desilike import utils
from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseLikelihood, BaseGaussianLikelihood 
from desilike.likelihoods.supernovae.base import BaseSNLikelihood

try:
    import jax.numpy as jnp
except Exception:
    jnp = np


class CompressedPantheonPlusSNLikelihood(BaseSNLikelihood):
    """
    Compressed Pantheon+ SN likelihood.

    Loads precomputed compressed arrays from an .npz file:
      - W                (nb, N)
      - z_full           (N,)
      - flatdata_binned0 (nb,)  (data vector with dM/Mb = 0 already applied)
      - cov_binned       (nb, nb)
      - (optional) ones_binned (nb,) = W @ 1

    Likelihood uses:
      flattheory_full(z_full) = 5*log10(DL(z_full)/h) + 25
      flattheory_binned       = W @ flattheory_full
      flatdata_binned         = flatdata_binned0 - dM * (W @ 1)
      cov_binned              fixed
    """
    name = "PantheonPlusSNCompressed"

    def initialize(self, *args, cosmo=None, compressed_fn=None, **kwargs):
        # Do NOT call BaseSNLikelihood.initialize (it would read full data)
        self.cosmo = cosmo

        if compressed_fn is None:
            raise ValueError("Provide compressed_fn pointing to an .npz with W/z_full/flatdata_binned0/cov_binned.")

        d = np.load(compressed_fn, allow_pickle=True)

        self.W = np.asarray(d["W"], dtype=float)
        self.z_full = np.asarray(d["z_full"], dtype=float)
        self.flatdata_binned0 = np.asarray(d["flatdata_binned0"], dtype=float)
        self.covariance = np.asarray(d["cov_binned"], dtype=float)

        nb, N = self.W.shape
        if self.z_full.size != N:
            raise ValueError(f"z_full has length {self.z_full.size} but W has N={N} columns.")
        if self.flatdata_binned0.size != nb:
            raise ValueError(f"flatdata_binned0 has length {self.flatdata_binned0.size} but W has nb={nb} rows.")
        if self.covariance.shape != (nb, nb):
            raise ValueError(f"cov_binned has shape {self.covariance.shape} but expected {(nb, nb)}.")

        # Precompute precision, std (used by BaseSNLikelihood.calculate)
        self.precision = utils.inv(self.covariance)
        self.std = np.sqrt(np.clip(np.diag(self.covariance), 0.0, np.inf))

        # W @ 1 (needed to propagate an additive constant dM/Mb correctly)
        if "ones_binned" in d:
            self.ones_binned = np.asarray(d["ones_binned"], dtype=float)
        else:
            self.ones_binned = self.W @ np.ones(N, dtype=float)

        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {"background": {"luminosity_distance": {"z": self.z_full}}}

    def calculate(self, dM=0.0, Mb=None):
        # allow Mb alias
        if Mb is not None:
            dM = Mb

        z = self.z_full

        # Theory at full z, then compress (consistent)
        flattheory_full = 5.0 * jnp.log10(self.cosmo.luminosity_distance(z) / self.cosmo["h"]) + 25.0
        self.flattheory = self.W @ flattheory_full

        # Compressed data with additive shift
        self.flatdata = self.flatdata_binned0 - float(dM) * self.ones_binned

        BaseSNLikelihood.calculate(self)