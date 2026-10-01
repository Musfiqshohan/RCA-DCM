"""
Evaluation metrics for comparing generated samples with true data.
Continuous variables only -- any discrete variable is expected to be
pre-processed (small noise added) into continuous form before it gets here.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance


def _plot_single_row(axes_row, gen_samples, true_dataset, var_names):
    """Helper: plot gen vs true density for each variable in one row of axes."""
    for idx, var_name in enumerate(var_names):
        ax = axes_row[idx]
        gen = gen_samples[var_name]
        true = true_dataset[var_name]
        gen_np = gen.cpu().numpy() if torch.is_tensor(gen) else np.array(gen)
        true_np = true.cpu().numpy() if torch.is_tensor(true) else np.array(true)
        gen_flat = gen_np.flatten().astype(float)
        true_flat = true_np.flatten().astype(float)
        min_len = min(len(gen_flat), len(true_flat))
        gen_flat, true_flat = gen_flat[:min_len], true_flat[:min_len]

        # gaussian_kde fails hard if there are NaNs/Infs (or too few points).
        # Filter non-finite values so plotting doesn't crash training.
        finite_mask = np.isfinite(gen_flat) & np.isfinite(true_flat)
        gen_flat = gen_flat[finite_mask]
        true_flat = true_flat[finite_mask]
        if len(gen_flat) < 2 or len(true_flat) < 2:
            ax.text(0.5, 0.5, "insufficient finite data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(var_name)
            continue
        x_min = min(gen_flat.min(), true_flat.min())
        x_max = max(gen_flat.max(), true_flat.max())
        x = np.linspace(x_min, x_max, 200)
        try:
            kde_gen = gaussian_kde(gen_flat)
            kde_true = gaussian_kde(true_flat)
            ax.plot(x, kde_true(x), label="true", color="C0", linewidth=2)
            ax.plot(x, kde_gen(x), label="gen", color="C1", linewidth=2)
        except ValueError as e:
            # Common causes: NaNs/Infs, all points identical (singular covariance), etc.
            ax.text(0.5, 0.5, f"kde failed: {e}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(var_name)
        ax.legend()


def plot_marginal_densities(nrm_samples, nrm_dat, anm_samples, anm_dat, out_dir, prc, plot_normal=True, plot_anomalous=True):
    """Plot gen vs true density for normal (top) and/or anomalous (bottom) in one figure."""
    var_names = list(nrm_samples.keys())
    n = len(var_names)
    ncols = n
    rows = []
    if plot_normal:
        rows.append(("normal", nrm_samples, nrm_dat))
    if plot_anomalous:
        rows.append(("anomalous", anm_samples, anm_dat))
    if not rows:
        return
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i, (label, samples, dat) in enumerate(rows):
        row_axes = np.atleast_1d(axes[i])
        _plot_single_row(row_axes, samples, dat, var_names)
        row_axes[0].set_ylabel(label)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{prc}_marginal_plot.png'), dpi=150, bbox_inches="tight")
    plt.close(fig)


def mmd_rbf(X, Y, gamma=1.0):
    """
    Maximum Mean Discrepancy (MMD) using RBF kernel.

    Args:
        X: numpy array of shape (n_samples, n_features)
        Y: numpy array of shape (n_samples, n_features)
        gamma: RBF kernel parameter

    Returns:
        MMD value (lower is better)
    """
    def rbf_kernel(X, Y, gamma):
        """Compute RBF kernel matrix."""
        XX = np.sum(X**2, axis=1).reshape(-1, 1)
        YY = np.sum(Y**2, axis=1).reshape(1, -1)
        XY = np.dot(X, Y.T)
        return np.exp(-gamma * (XX - 2*XY + YY))

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return round(mmd, 4)


def _to_numpy(x):
    """Convert torch tensor or array-like object to numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def evaluate_cont_loss(gen_samples, true_dataset, prc, eps=1e-8, gamma=1.0):
    """
    Evaluate a single continuous variable (prc) against its true data.

    Metrics:
        - MMD
        - Wasserstein distance
        - BARO-style robust score

    Args:
        gen_samples: dict {var_name: tensor or array}
        true_dataset: dataset object with true data
        prc: the one variable (node under test) to score -- this used to loop
            over every variable in gen_samples and score all of them, but
            tvd/_compute_scores only ever read the prc column back out
            (PERF(A1-18), now moot: nothing but prc is computed).
        eps: small value to avoid division by zero
        gamma: RBF kernel parameter for MMD

    Returns:
        dict with a single float per metric: {"mmd": ..., "wasserstein": ..., "baro": ...}
    """
    gen_data = _to_numpy(gen_samples[prc]).flatten().astype(float)
    true_data = _to_numpy(true_dataset[prc]).flatten().astype(float)

    min_len = min(len(gen_data), len(true_data))
    gen_data = gen_data[:min_len]
    true_data = true_data[:min_len]

    gen_data_2d = gen_data.reshape(-1, 1)
    true_data_2d = true_data.reshape(-1, 1)

    # MMD
    mmd_value = round(float(mmd_rbf(gen_data_2d, true_data_2d, gamma=gamma)), 4)

    # Wasserstein distance
    wd = round(float(wasserstein_distance(true_data, gen_data)), 4)

    # BARO-style robust score
    true_median = np.median(true_data)
    q75 = np.quantile(true_data, 0.75)
    q25 = np.quantile(true_data, 0.25)
    true_iqr = q75 - q25

    scale = true_iqr if true_iqr > eps else np.std(true_data) + eps

    baro_score = round(float(np.max(np.abs(gen_data - true_median) / scale)), 4)

    return {
        "mmd": mmd_value,
        "wasserstein": wd,
        "baro": baro_score,
    }
