import numpy as np
from cosmoprimo import Cosmology
from scipy.linalg import block_diag

# Speed of light in km/s
c_l_kms = 299792.458

class BAOCosmology:
    """
    A helper class that wraps a Cosmology instance to provide BAO distance methods.
    """
    def __init__(self, cosmo=None, engine='camb'):
        self.engine = engine
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = Cosmology(w0_fld=-1, wa_fld=0, Omega_m=0.3,
                                   omega_b=0.022, h=0.7)
            self.cosmo.set_engine(engine)

    def compute_DMoverRs(self, z):
        return self.cosmo.comoving_angular_distance(z) / self.cosmo.rs_drag

    def compute_DHoverRs(self, z):
        return c_l_kms / (self.cosmo.efunc(z)*100) / self.cosmo.rs_drag

    def compute_DVoverRs(self, z):
        DM = self.compute_DMoverRs(z)
        DH = self.compute_DHoverRs(z)
        return (z * DM**2 * DH)**(1.0/3.0)

class BAOLikelihood:
    # Class-level cache for BAO data; loaded only once per dataset.
    _data_loaded = {}
    _data = {}

    @classmethod
    def load_default_data(cls, data_dir=".", dataset="DESI"):
        """
        Load BAO data        
        For dataset 'DESI', loads from "DESI_bao_data.npz".
        For dataset 'DESI_SDSS', loads from "DESI_SDSS_bao_data.npz".
        """
        # Check if already loaded for this dataset.
        if dataset in cls._data_loaded and cls._data_loaded[dataset]:
            return

        if dataset.upper() == "DESI":
            filename = f"{data_dir}/DESI_bao_data.npz"
        elif dataset.upper() == "DESI_SDSS":
            filename = f"{data_dir}/DESI_SDSS_bao_data.npz"
        else:
            raise ValueError("Invalid dataset. Use 'DESI' or 'DESI_SDSS'.")
        
        loaded = np.load(filename, allow_pickle=True)
        cls._data[dataset] = {
            "data_vector": loaded["data"],
            "covmat": loaded["cov"],
            "redshifts": loaded["zeff"],
            "types": loaded["types"]
        }
        cls._data_loaded[dataset] = True

    def __init__(self, cosmo=None, data_dir="bao_data", dataset="DESI", engine='camb'):
        BAOLikelihood.load_default_data(data_dir=data_dir, dataset=dataset)
        data = BAOLikelihood._data[dataset]
        self.data_vector = data["data_vector"]
        self.covmat = data["covmat"]
        self.redshifts = data["redshifts"]
        self.types = data["types"]
        if cosmo is not None:
            if not hasattr(cosmo, "compute_DMoverRs"):
                self.model = BAOCosmology(cosmo=cosmo, engine=engine)
            else:
                self.model = cosmo
        else:
            self.model = BAOCosmology(engine=engine)
        self.loglikelihood = None

    def calculate(self, sys_coeff=None):
        """
        Compute the BAO log-likelihood.        
        If sys_coeff is provided, scales the covariance matrix to account for systematics.
        """
        theory_vector = np.empty_like(self.data_vector, dtype=float)
        for typ in np.unique(self.types):
            indices = np.where(self.types == typ)[0]
            z_vals = self.redshifts[indices]
            if typ == "DM_over_rs":
                theory_vals = self.model.compute_DMoverRs(z_vals)
            elif typ == "DH_over_rs":
                theory_vals = self.model.compute_DHoverRs(z_vals)
            elif typ == "DV_over_rs":
                theory_vals = self.model.compute_DVoverRs(z_vals)
            else:
                raise ValueError(f"Invalid observable type: {typ}")
            theory_vector[indices] = theory_vals

        if sys_coeff is None:
            delta = self.data_vector - theory_vector
            inv_cov = np.linalg.inv(self.covmat)
            chi2 = delta.T @ inv_cov @ delta
            self.loglikelihood = -0.5 * chi2
        else:
            new_cov = sys_coeff * self.covmat
            inv_cov = np.linalg.inv(new_cov)
            logdet = np.log(np.linalg.det(new_cov))
            delta = self.data_vector - theory_vector
            chi2 = delta.T @ inv_cov @ delta
            self.loglikelihood = -0.5 * (chi2 + logdet)
        return self.loglikelihood
