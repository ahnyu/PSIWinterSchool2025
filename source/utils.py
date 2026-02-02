import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

def plot_sn_rebin_with_residual(z, mu, cov, z_b, mu_b, cov_b, W, show=True, lim_y=None):

    z = np.asarray(z, float)
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)

    z_b = np.asarray(z_b, float)
    mu_b = np.asarray(mu_b, float)
    cov_b = np.asarray(cov_b, float)
    W = np.asarray(W, float)

    err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    err_b = np.sqrt(np.clip(np.diag(cov_b), 0.0, np.inf))

    s = np.argsort(z)
    sb = np.argsort(z_b)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    ax0.errorbar(z[s], mu[s], yerr=err[s], fmt=".", ms=3, lw=0.8, alpha=0.5, label="Original")
    ax0.errorbar(z_b[sb], mu_b[sb], yerr=err_b[sb], fmt="o", ms=5, lw=1.2, label="Rebinned")
    ax0.set_ylabel(r"$\mu$")
    ax0.grid(True, alpha=0.3)
    if lim_y:
        ax0.set_ylim(lim_y)
    ax0.legend()

    mu_b_from_W = W @ mu
    resid = mu_b - mu_b_from_W
    ax1.axhline(0.0, lw=1.0)
    ax1.plot(z_b[sb], resid[sb], "o", ms=4)
    ax1.set_xlabel("z")
    ax1.set_ylabel(r"$\Delta\mu$")
    ax1.grid(True, alpha=0.3)

    if show:
        plt.show()
    return fig, (ax0, ax1)

def load_chain(fn, burn=0.0):
    data = np.loadtxt(fn)
    if data.ndim == 1:
        data = data[None, :]

    ncols = data.shape[1]
    if ncols < 4:
        raise ValueError(f"{fn}: expected >= 4 columns, got {ncols}")

    npar = ncols - 3
    samples = data[:, :npar]
    weights = data[:, npar]
    # logl = data[:, npar + 1]  # not needed for plotting
    # logp = data[:, npar + 2]

    # burn can be fraction (0<burn<1) or integer (>=1)
    if burn:
        if 0 < burn < 1:
            ib = int(burn * samples.shape[0])
        else:
            ib = int(burn)
        samples = samples[ib:]
        weights = weights[ib:]

    # normalize weights (GetDist is fine either way, but this helps numerical stability)
    wsum = np.sum(weights)
    if wsum > 0:
        weights = weights / wsum

    return samples, weights, npar


def default_names_labels(model, npar):
    if model.lower() == "w0wacdm":
        names = ["w0", "wa", "Omega_m", "omega_b", "h", "dM"]
        labels = [r"w_0", r"w_a", r"\Omega_m", r"\omega_b", r"h", r"\Delta M"]
    elif model.lower() == "lcdm":
        names = ["Omega_m", "omega_b", "h", "dM"]
        labels = [r"\Omega_m", r"\omega_b", r"h", r"\Delta M"]
    else:
        raise ValueError("Unknown model. Use w0waCDM or LCDM.")

    if len(names) != npar:
        raise ValueError(f"Model {model} expects {len(names)} params, but chain has {npar}. "
                         f"Use --names to override.")
    return names, labels


def make_mcsamples(samples, weights, names, labels, label):
    return MCSamples(
        samples=samples,
        weights=weights,
        names=names,
        labels=labels,
        label=label,
    )


def print_quick_stats(mc, names, title):
    stats = mc.getMargeStats()
    print(f"\n=== {title} ===")
    for n in names:
        p = stats.parWithName(n)
        mean = p.mean
        sd = p.sigma
        lo = p.limits[0].lower
        hi = p.limits[0].upper
        print(f"{n:10s}: mean={mean: .5f}  sigma={sd: .5f}  68%=[{lo: .5f}, {hi: .5f}]")
    print("================\n")
