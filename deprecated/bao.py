import numpy as np
from cosmoprimo import Cosmology

# Speed of light in km/s
c_l_kms = 299792.458

class BAOCosmology:
    def __init__(self, engine='camb'):
        self.engine = engine
        self.cosmo = Cosmology(w0_fld=-1, wa_fld=0, Omega_m=0.3,
                               omega_b=0.022, h=0.7)
        self.cosmo.set_engine(engine)
    
    def update_parameters(self, w0, wa, Omega_m, omega_b, h):
        self.cosmo = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m,
                               omega_b=omega_b, h=h)
        self.cosmo.set_engine(self.engine)
    
    def compute_DMoverRs(self, z):
        return self.cosmo.comoving_angular_distance(z) / self.cosmo.rs_drag
    
    def compute_DHoverRs(self, z):
        return c_l_kms / (self.cosmo.efunc(z) * 100) / self.cosmo.rs_drag
    
    def compute_DVoverRs(self, z):
        DM = self.compute_DMoverRs(z)
        DH = self.compute_DHoverRs(z)
        return (z * DM**2 * DH) ** (1.0/3.0)


class BAOLikelihood:
    def __init__(self, data_vector, covmat, redshifts, types, engine='camb'):
        self.data_vector = np.array(data_vector)
        self.covmat = covmat
        self.redshifts = np.array(redshifts)
        self.types = np.array(types)
        self.model = BAOCosmology(engine=engine)
    
    def log_likelihood(self, params):

        w0,wa,Omega_m,omega_b,h=params
        
        if w0 + wa >= 0:
            return -np.inf
        if Omega_m * h**2 <= omega_b:
            return -np.inf
            
        self.model.update_parameters(w0,wa,Omega_m,omega_b,h)
        
        theory_vector = np.empty_like(self.data_vector, dtype=float)
        
        unique_types = np.unique(self.types)
        for typ in unique_types:

            indices = np.where(self.types == typ)[0]

            z_vals = self.redshifts[indices]
            
            # Compute the theory value for these redshifts using the appropriate method.
            if typ == 'DM_over_rs':
                theory_vals = self.model.compute_DMoverRs(z_vals)
            elif typ == 'DH_over_rs':
                theory_vals = self.model.compute_DHoverRs(z_vals)
            elif typ == 'DV_over_rs':
                theory_vals = self.model.compute_DVoverRs(z_vals)
            else:
                raise ValueError("Invalid type: {}. Allowed types: DM_over_rs, DH_over_rs, DV_over_rs.".format(typ))
            
            theory_vector[indices] = theory_vals

        #print('theory_vector:',theory_vector)

        delta = self.data_vector - theory_vector
        inv_cov = np.linalg.inv(self.covmat)
        chi2 = delta.T @ inv_cov @ delta
        
        return -0.5 * chi2


class BAOLikelihood_sys:
    def __init__(self, data_vector, covmat, redshifts, types, engine='camb'):
        
        self.data_vector = np.array(data_vector)
        self.covmat = covmat
        self.redshifts = np.array(redshifts)
        self.types = np.array(types)
        self.model = BAOCosmology(engine=engine)
    
    def log_likelihood(self, params):

        w0, wa, Omega_m, omega_b, h, sys_coeff = params
        
        if w0 + wa >= 0:
            return -np.inf
        if Omega_m * h**2 <= omega_b:
            return -np.inf
            
        self.model.update_parameters(w0, wa, Omega_m, omega_b, h)

        theory_vector = np.empty_like(self.data_vector, dtype=float)

        unique_types = np.unique(self.types)
        for typ in unique_types:
            indices = np.where(self.types == typ)[0]
            z_vals = self.redshifts[indices]
            
            if typ == 'DM_over_rs':
                theory_vals = self.model.compute_DMoverRs(z_vals)
            elif typ == 'DH_over_rs':
                theory_vals = self.model.compute_DHoverRs(z_vals)
            elif typ == 'DV_over_rs':
                theory_vals = self.model.compute_DVoverRs(z_vals)
            else:
                raise ValueError("Invalid type: {}. Allowed types: DM_over_rs, DH_over_rs, DV_over_rs.".format(typ))
            
            theory_vector[indices] = theory_vals

        new_cov = sys_coeff * self.covmat

        inv_cov = np.linalg.inv(new_cov)
        logdet = np.log(np.linalg.det(new_cov))

        delta = self.data_vector - theory_vector
        chi2 = delta.T @ inv_cov @ delta
        logL = -0.5 * (chi2 + logdet)
        
        return logL