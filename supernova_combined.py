import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from astropy.table import Table
import fitsio

def rebin_SN(z, mu, cov, bins, project=True):
    """
    Rebin SN distance moduli into wider redshift bins using full (non-diagonal) covariance.

    Parameters
    ----------
    z : array (N,)
        Redshifts of individual SNe.
    mu : array (N,)
        Distance moduli of individual SNe.
    cov : array (N, N)
        Full covariance matrix of mu (NOT its inverse).
    bins : array (Nbins+1,)
        Redshift bin edges.
    project : bool, optional
        If True, project out a global offset mode using the full covariance:
        - Build weights w ~ C^{-1} 1 / (1^T C^{-1} 1)
        - Apply projector P = I - 1 w^T to both mu and cov.

    Returns
    -------
    z_binned : array (Nbins,)
        Weighted mean redshift in each bin.
    mu_binned : array (Nbins,)
        Optimally weighted mean mu in each bin (after optional projection).
    err_binned : array (Nbins,)
        1-sigma error for each binned mu (sqrt of diagonal of cov_binned).
    cov_binned : array (Nbins, Nbins)
        Full covariance of the binned mu vector.
    """
    z = np.asarray(z)
    mu = np.asarray(mu)
    cov = np.asarray(cov)

    N = len(z)
    assert cov.shape == (N, N)

    # ----------------------------------------------------
    # 1. Project out global offset using full C^{-1}
    # ----------------------------------------------------
    if project:
        icov_full = np.linalg.inv(cov)
        ones = np.ones(N)

        # w = C^{-1} 1 / (1^T C^{-1} 1), so sum_j w_j = 1
        w = icov_full @ ones
        norm = ones @ w
        w /= norm

        # Projector P = I - 1 w^T, so P 1 = 0 and P^2 = P
        P = np.eye(N) - np.outer(ones, w)

        # Apply to data *and* covariance
        mu = P @ mu
        cov = P @ cov @ P.T

    # ----------------------------------------------------
    # 2. Build binning matrix using full covariance
    # ----------------------------------------------------
    centers = 0.5 * (bins[1:] + bins[:-1])
    nbins = len(centers)

    W = np.zeros((nbins, N))    # binning / compression matrix
    err_binned = np.zeros(nbins)

    for i in range(nbins):
        mask = (z >= bins[i]) & (z < bins[i+1])
        idx = np.where(mask)[0]
        if idx.size == 0:
            # No SNe in this bin; you can choose to raise instead
            continue

        C_sub = cov[np.ix_(idx, idx)]
        icov_sub = np.linalg.inv(C_sub)

        ones_sub = np.ones(idx.size)

        # Optimal weights for the mean in this bin:
        # w_bin = C_sub^{-1} 1 / (1^T C_sub^{-1} 1)
        w_bin = icov_sub @ ones_sub
        norm_bin = ones_sub @ w_bin
        w_bin /= norm_bin

        W[i, idx] = w_bin

        # Var(mean) = 1 / (1^T C_sub^{-1} 1)
        err_binned[i] = np.sqrt(1.0 / norm_bin)

    # ----------------------------------------------------
    # 3. Compress z, mu, and cov
    # ----------------------------------------------------
    z_binned = W @ z
    mu_binned = W @ mu
    cov_binned = W @ cov @ W.T

    return z_binned, mu_binned, err_binned, cov_binned

# example script to run them, set geometry_distance_dir to data folder dir
SN_data = {}
geometry_distance_dir = Path('/home/eleannakolonia/cosmo_project/venv/PSI_winter_school/SN_data')

bins = np.array([0.01, 0.125, 0.235, 0.325, 0.425, 0.625, 0.825, 3])

data = Table.read(geometry_distance_dir / 'Pantheon+SH0ES.dat', format='ascii')
cov = np.loadtxt(geometry_distance_dir / 'Pantheon+SH0ES_STAT+SYS.cov', skiprows=1).reshape((len(data), len(data)))
mu = data['MU_SH0ES'] #- 5 * np.log10((1 + data['zHEL']) / (1 + data['zHD']))
SN_data['pantheonplus'] = (bins,) + rebin_SN(data['zHD'], mu , cov, bins, project=False)

fig, ax = plt.subplots(ncols=1, nrows=1, sharex=False, sharey=False, figsize=(8, 6))
ax.errorbar(SN_data['pantheonplus'][1],SN_data['pantheonplus'][2],yerr=SN_data['pantheonplus'][3])
ax.plot(data['zHD'],mu,ls='',marker='o')
# print(mu)
# print(data)

data = fitsio.read(geometry_distance_dir / 'mu_mat_union3_cosmo=2_mu.fits')
# Union3 incov is already marginalized over an offset in mu
SN_data['union3'] = (bins,) + rebin_SN(data[0, 1:], data[1:, 0], np.linalg.inv(data[1:, 1:]), bins, project=False)

# load DES Y5 SN data
data = pd.read_csv(f'{geometry_distance_dir}/DES-SN5YR_HD.csv')
cov = np.loadtxt(geometry_distance_dir / 'DESY5_STAT+SYS.txt', skiprows=1).reshape((len(data), len(data)))
cov = cov + np.diag(data['MUERR_FINAL']**2)
mu = data['MU'] #- 5 * np.log10((1 + data['zHEL']) / (1 + data['zHD']))
SN_data['desy5sn'] = (bins,) + rebin_SN(data['zHD'], mu, cov, bins, project=False)

# print(SN_data)

np.save("SN_dataset.npy", SN_data, allow_pickle=True)


class SN_Combined:

    def __init__(self, z, mu, cov, name=None):
        self.name = name
        self.z = np.asarray(z)
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.icov = np.linalg.inv(self.cov)

        self.loglikelihood = None
    
    def predict_mu(self, cosmo=None):
        """
        Predicts the distance modulus (mu) for a given redshift (z) using a specified cosmological model.

        Parameters:
        z (float or array-like): Redshift(s) at which to predict the distance modulus.
        cosmo (object, optional): Cosmological model to use for the prediction. If None, the DESI fiducial model is used.

        Returns:
        dict: A dictionary containing the following keys:
            - 'mu': The predicted distance modulus for the given cosmological model.
            - 'mu_fid': The predicted distance modulus for the fiducial DESI model.
            - 'delta_mu': The difference between the predicted distance modulus and the fiducial distance modulus.
        """
        from cosmoprimo.fiducial import DESI
        from cosmoprimo import constants
        fiducial = DESI()
        if cosmo is None:
            cosmo = DESI()

        def predict(cosmo): 
            if isinstance(cosmo, tuple):
                hrdrag = cosmo[1]
                DL = cosmo[0].luminosity_distance(self.z)
            else:
                DL = cosmo.luminosity_distance(self.z)
            mu_th = 5*np.log10(DL)+25
            return DL, mu_th

        DL, mu_th = predict(cosmo)
        DL_fid, mu_fid = predict(fiducial)
        return {'mu_th': mu_th, 'mu_fid': mu_fid, 'delta_mu': mu_th - mu_fid}
    
    def calculate(self, cosmo, dM=None, sys_coeff=None):
        if cosmo is None:
            raise ValueError("Cosmology not set.")
        

        out = self.predict_mu(cosmo)
        mu_th = out["mu_th"]

        resid = self.mu - (mu_th + dM) 
        chi2 = resid @ self.icov @ resid
        if sys_coeff == None:
            self.loglikelihood = -0.5 * chi2
        else:
            chi2_sys = chi2 / sys_coeff
            cov_sys = sys_coeff * cov
            self.loglikelihood = -0.5 * (chi2_sys + np.log(np.linalg.det(cov_sys)))
        return self.loglikelihood