import numpy as np
from cosmoprimo import Cosmology

class CMBCompressedLikelihood:
    # Class-level cache for CMB data; loaded only once.
    _data_loaded = False
    _data = {}

    @classmethod
    def load_default_data(cls):
        """
        Loads default compressed CMB data.
        """
        cls._data["data_vector"] = np.array([0.01041027, 0.02223208, 0.14207901])
        cls._data["covmat"] = np.array([
            [6.62099420e-12,  1.24442058e-10, -1.19287532e-09],
            [1.24442058e-10,  2.13441666e-08, -9.40008323e-08],
            [-1.19287532e-09, -9.40008323e-08,  1.48841714e-06]
        ])
        cls._data_loaded = True

    def __init__(self, engine='camb', cosmo=None):
        if not CMBCompressedLikelihood._data_loaded:
            CMBCompressedLikelihood.load_default_data()
        self.data_vector = CMBCompressedLikelihood._data["data_vector"]
        self.covmat = CMBCompressedLikelihood._data["covmat"]
        self.inv_cov = np.linalg.inv(self.covmat)
        self.engine = engine
        if cosmo is not None:
            self.model = cosmo
        else:
            self.model = Cosmology(w0_fld=-1, wa_fld=0, Omega_m=0.3,
                                    omega_b=0.022, h=0.7)
            self.model.set_engine(engine)

    def calculate(self):
        """
        Compute the compressed CMB log-likelihood.        
        The theory vector is defined as:
        [ theta_star, omega_b, omega_bc]
        """
        theta_star = self.model.theta_star
        h = self.model.h
        omega_b = self.model.Omega0_b * h**2
        omega_bc = omega_b + self.model.Omega0_cdm * h**2
        theory_vector = np.array([theta_star, omega_b, omega_bc])
        delta = self.data_vector - theory_vector
        chi2 = delta.T @ self.inv_cov @ delta
        return -0.5 * chi2
