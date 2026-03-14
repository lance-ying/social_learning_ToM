"""Shared plotting and CI utilities for 4-panel correlation figures."""

import numpy as np
from scipy import stats


def pearson_statistic(sample1, sample2, axis=-1):
    """Return Pearson r for paired samples."""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    if sample1.ndim == 1:
        return _pearson_scalar(sample1, sample2)

    sample1 = np.moveaxis(sample1, axis, -1)
    sample2 = np.moveaxis(sample2, axis, -1)
    out = np.empty(sample1.shape[:-1], dtype=float)

    for idx in np.ndindex(sample1.shape[:-1]):
        out[idx] = _pearson_scalar(sample1[idx], sample2[idx])
    return out


def _pearson_scalar(sample1, sample2):
    """Scalar Pearson r for one paired sample."""
    x = np.asarray(sample1)
    y = np.asarray(sample2)
    return float(stats.pearsonr(x, y)[0])


def bootstrap_r_ci(
    x,
    y,
    n_resamples=1000,
    confidence_level=0.95,
    seed=42,
    method="percentile",
):
    """Compute Pearson r and bootstrap confidence interval."""
    x = np.asarray(x)
    y = np.asarray(y)
    r = _pearson_scalar(x, y)
    rng = np.random.default_rng(seed)
    kwargs = {
        "paired": True,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "method": method,
    }

    try:
        res = stats.bootstrap((x, y), _pearson_scalar, rng=rng, **kwargs)
    except TypeError:
        # Compatibility with older SciPy bootstrap signature.
        res = stats.bootstrap((x, y), _pearson_scalar, random_state=seed, **kwargs)

    return r, float(res.confidence_interval.low), float(res.confidence_interval.high)


def apply_reference_style(ax):
    """Apply style matching the requested reference figure."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=20, colors="#1a1a1a")


def plot_points_errorbars_and_fit(ax, x, y, yerr):
    """Plot scatter points with y-error bars and a linear best-fit line."""
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        alpha=0.9,
        capsize=0,
        markersize=7,
        elinewidth=1.6,
        color="#8dc5df",
        ecolor="#c8ddea",
        markeredgewidth=0.0,
    )

    if len(x) >= 2:
        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, fit_fn(x_line), color="#e79a9a", linewidth=2.0)


def annotate_r_ci(ax, r, ci_low, ci_high):
    """Add correlation and CI annotation in top-left of axis."""
    ax.text(
        0.05,
        0.90,
        f"r = {r:.2f}\nCI = [{ci_low:.2f}, {ci_high:.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        color="#1a1a1a",
    )
