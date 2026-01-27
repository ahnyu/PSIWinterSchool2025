import numpy as np
from cosmoprimo import Cosmology
from scipy.linalg import cho_factor, cho_solve

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
    _data_loaded = {}
    _data = {}

    @classmethod
    def load_default_data(cls, data_dir=".", dataset="DESI"):
        key = dataset.upper()
        if cls._data_loaded.get(key, False):
            return
        if key == "DESI":
            filename = f"{data_dir}/DESI_bao_data.npz"
        elif key == "DESI_SDSS":
            filename = f"{data_dir}/DESI_SDSS_bao_data.npz"
        else:
            raise ValueError("Invalid dataset. Use 'DESI' or 'DESI_SDSS'.")
        loaded = np.load(filename, allow_pickle=True)
        cls._data[key] = dict(
            data_vector=np.asarray(loaded["data"], dtype=float),
            covmat=np.asarray(loaded["cov"], dtype=float),
            redshifts=np.asarray(loaded["zeff"], dtype=float),
            types=np.asarray(loaded["types"]).astype(str),
        )
        cls._data_loaded[key] = True

    def __init__(self, cosmo=None, data_dir="bao_data", dataset="DESI", engine="camb"):
        self.load_default_data(data_dir=data_dir, dataset=dataset)
        data = self._data[dataset.upper()]

        self.data_vector = data["data_vector"]
        self.covmat = data["covmat"]
        self.redshifts = data["redshifts"]
        self.types = data["types"]

        # Precompute type and indices
        self._idx = {t: np.where(self.types == t)[0] for t in np.unique(self.types)}
        self.ndata = self.data_vector.size

        # Precompute Cholesky and logdet(C)
        self._cho = cho_factor(self.covmat, lower=True, check_finite=False)
        L = self._cho[0]
        self._logdetC = 2.0 * np.sum(np.log(np.diag(L)))

        if cosmo is not None:
            self.model = cosmo if hasattr(cosmo, "compute_DMoverRs") else BAOCosmology(cosmo=cosmo, engine=engine)
        else:
            self.model = BAOCosmology(engine=engine)

    def _theory_vector(self):
        th = np.empty_like(self.data_vector, dtype=float)
        for typ, inds in self._idx.items():
            z = self.redshifts[inds]
            if typ == "DM_over_rs":
                th[inds] = self.model.compute_DMoverRs(z)
            elif typ == "DH_over_rs":
                th[inds] = self.model.compute_DHoverRs(z)
            elif typ == "DV_over_rs":
                th[inds] = self.model.compute_DVoverRs(z)
            else:
                raise ValueError(f"Invalid observable type: {typ}")
        return th

    def calculate(self, sys_coeff=None):
        th = self._theory_vector()
        delta = self.data_vector - th

        chi2_0 = float(delta @ cho_solve(self._cho, delta, check_finite=False))

        if sys_coeff is None:
            return -0.5 * chi2_0
        s = float(sys_coeff)
        if not np.isfinite(s) or s <= 0.0:
            return -np.inf

        chi2 = chi2_0 / s
        logdet = self._logdetC + self.ndata * np.log(s)
        return -0.5 * (chi2 + logdet)
