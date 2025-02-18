import numpy as np
from cosmoprimo import Cosmology
class CMBCompressedLikelihood:
    def __init__(self, data_vector, covmat, engine='camb'):
        self.data_vector = np.array(data_vector)
        self.covmat = np.array(covmat)
        self.inv_cov = np.linalg.inv(self.covmat)
        self.engine = engine
        self.model = Cosmology(w0_fld=-1, wa_fld=0, Omega_m=0.3, omega_b=0.022, h=0.7)
        self.model.set_engine(engine)
    
    def log_likelihood(self, params):

        w0, wa, Omega_m, omega_b, h = params

        if w0 + wa >= 0:
            return -np.inf
        if Omega_m * h**2 <= omega_b:
            return -np.inf
        self.model = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m, omega_b=omega_b, h=h)
        self.model.set_engine(self.engine)
        theta_star = self.model.theta_star
        omega_bc = omega_b+self.model.Omega0_cdm*h**2
        theory_vector = np.array([theta_star,omega_b,omega_bc])
        
        # Compute the residual vector.
        delta = self.data_vector - theory_vector
        chi2 = delta.T @ self.inv_cov @ delta
        
        # Return the Gaussian log-likelihood (ignoring constant terms).
        return -0.5 * chi2