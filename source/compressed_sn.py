import numpy as np

from desilike import utils
from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseLikelihood, BaseGaussianLikelihood 
from desilike.likelihoods.supernovae.base import BaseSNLikelihood

try:
    import jax.numpy as jnp
except Exception:
    jnp = np

def rebin_SN(z, mu, cov, bins, *, drop_empty=True, include_rightmost=True,
             return_W=False):

    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    bins = np.asarray(bins, dtype=float)

    N = z.size
    if mu.size != N:
        raise ValueError(f"mu has length {mu.size} but z has length {N}")
    if cov.shape != (N, N):
        raise ValueError(f"cov has shape {cov.shape} but expected {(N, N)}")
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("bins must be 1D array of bin edges with length >= 2")

    nbins = bins.size - 1

    W = np.zeros((nbins, N), dtype=float)
    keep = np.ones(nbins, dtype=bool)

    for i in range(nbins):
        lo, hi = bins[i], bins[i + 1]

        if i == nbins - 1 and include_rightmost:
            mask = (z >= lo) & (z <= hi)
        else:
            mask = (z >= lo) & (z < hi)

        idx = np.where(mask)[0]
        if idx.size == 0:
            keep[i] = False
            continue

        C = cov[np.ix_(idx, idx)]

        ones = np.ones(idx.size, dtype=float)
        try:
            cf = cho_factor(C, lower=True, check_finite=False)
            x = cho_solve(cf, ones, check_finite=False)  # x = C^{-1} 1
        except Exception:
            x = np.linalg.lstsq(C, ones, rcond=None)[0]

        norm = float(ones @ x)  # 1^T C^{-1} 1
        if not np.isfinite(norm) or norm <= 0.0:
            keep[i] = False
            continue

        w = x / norm
        W[i, idx] = w

    if drop_empty:
        W = W[keep]
    else:
        pass

    mu_binned = W @ mu
    z_binned = W @ z
    cov_binned = W @ cov @ W.T

    cov_binned = 0.5 * (cov_binned + cov_binned.T)

    diag = np.diag(cov_binned)
    diag = np.where(diag < 0, 0.0, diag)
    err_binned = np.sqrt(diag)

    if return_W:
        return z_binned, mu_binned, err_binned, cov_binned, W
    return z_binned, mu_binned, err_binned, cov_binned


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
        self.cosmo = cosmo

        if compressed_fn is None:
            raise ValueError("Provide compressed_fn pointing to an .npz")

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

        self.precision = utils.inv(self.covariance)
        self.std = np.sqrt(np.clip(np.diag(self.covariance), 0.0, np.inf))

        if "ones_binned" in d:
            self.ones_binned = np.asarray(d["ones_binned"], dtype=float)
        else:
            self.ones_binned = self.W @ np.ones(N, dtype=float)

        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {"background": {"luminosity_distance": {"z": self.z_full}}}

    def calculate(self, dM=0.0, Mb=None):
        if Mb is not None:
            dM = Mb

        z = self.z_full

        flattheory_full = 5.0 * jnp.log10(self.cosmo.luminosity_distance(z) / self.cosmo["h"]) + 25.0
        self.flattheory = self.W @ flattheory_full

        self.flatdata = self.flatdata_binned0 - float(dM) * self.ones_binned

        BaseSNLikelihood.calculate(self)


class CompressedUnion3SNLikelihood(BaseSNLikelihood):
    """
    Compressed Union3 SN likelihood.

    Loads precomputed compressed arrays from an .npz file:
      - W                (nb, N)
      - z_full           (N,)      (Union3 uses 'zcmb')
      - flatdata_binned0 (nb,)     (compressed data vector with dM=0)
      - cov_binned       (nb, nb)
      - (optional) ones_binned (nb,) = W @ 1

    Likelihood uses:
      flattheory_full(z_full) = 5*log10(100 * DL(z_full)) + 25
      flattheory_binned       = W @ flattheory_full
      flatdata_binned         = flatdata_binned0 - dM * (W @ 1)
    """
    name = "Union3SNCompressed"

    def initialize(self, *args, cosmo=None, compressed_fn=None, **kwargs):
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

        self.precision = utils.inv(self.covariance)
        self.std = np.sqrt(np.clip(np.diag(self.covariance), 0.0, np.inf))

        if "ones_binned" in d:
            self.ones_binned = np.asarray(d["ones_binned"], dtype=float)
        else:
            self.ones_binned = self.W @ np.ones(N, dtype=float)

        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {"background": {"luminosity_distance": {"z": self.z_full}}}

    def calculate(self, dM=0.0):
        z = self.z_full
        
        # Union3 theory: 5*log10(100 * DL) + 25
        # Note: cosmoprimo returns DL in [Mpc/h], so 100*DL is dimensionless H0*d_L
        flattheory_full = 5.0 * jnp.log10(100.0 * self.cosmo.luminosity_distance(z)) + 25.0
        self.flattheory = self.W @ flattheory_full

        self.flatdata = self.flatdata_binned0 - float(dM) * self.ones_binned

        BaseSNLikelihood.calculate(self)

class CompressedDESY5SNLikelihood(BaseSNLikelihood):

    name = "DESY5SNCompressed"

    def initialize(self, *args, cosmo=None, compressed_fn=None, **kwargs):
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

        self.precision = utils.inv(self.covariance)
        self.std = np.sqrt(np.clip(np.diag(self.covariance), 0.0, np.inf))

        if "ones_binned" in d:
            self.ones_binned = np.asarray(d["ones_binned"], dtype=float)
        else:
            self.ones_binned = self.W @ np.ones(N, dtype=float)

        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {"background": {"luminosity_distance": {"z": self.z_full}}}

    def calculate(self, Mb=0.0):
        z = self.z_full

        flattheory_full = 5.0 * jnp.log10(self.cosmo.luminosity_distance(z) / self.cosmo["h"]) + 25.0
        self.flattheory = self.W @ flattheory_full

        self.flatdata = self.flatdata_binned0 - float(Mb) * self.ones_binned

        BaseSNLikelihood.calculate(self)