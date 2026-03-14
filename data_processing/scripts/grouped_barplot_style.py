#!/usr/bin/env python3
"""Shared style helpers for grouped model barplots across experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = ["exp1", "exp2", "exp3", "exp4"]
EXPERIMENT_LABELS = ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"]
SERIES = [
    ("Human", "human"),
    ("Rational Mentalizing", "full_model"),
    ("Social Mentalizing", "social_mentalizing"),
    ("Rational Non-Mentalizing", "rational_non_mentalizing"),
    ("Naive Observer", "naive_observer"),
]
SERIES_TICK_LABELS = [
    "Human",
    "Rational\nMentalizing",
    "Social\nMentalizing",
    "Rational Non-\nMentalizing",
    "Naive\nObserver",
]
EXPERIMENT_COLORS = {
    "exp1": "#444444",
    "exp2": "#c95f5f",
    "exp3": "#d98e3d",
    "exp4": "#5b8fd1",
}

BAR_WIDTH = 0.15
BAR_ALPHA = 0.9
CAPSIZE = 4
ERROR_COLOR = "#333333"
FIGSIZE_SINGLE = (10, 7)
FIGSIZE_MULTI_PANEL = (8, 7)
TITLE_FONTSIZE = 16
YLABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11


def apply_axis_style(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_xticks(np.arange(len(SERIES)))
    ax.set_xticklabels(SERIES_TICK_LABELS, fontsize=TICK_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def plot_grouped_bars(ax: plt.Axes, means_by_series: list[list[float]], sds_by_series: list[list[float]]) -> None:
    x = np.arange(len(SERIES))
    for exp_idx, (exp_label, exp_key) in enumerate(zip(EXPERIMENT_LABELS, EXPERIMENTS)):
        offset = (exp_idx - (len(EXPERIMENTS) - 1) / 2) * BAR_WIDTH
        means = [series_means[exp_idx] for series_means in means_by_series]
        sds = [series_sds[exp_idx] for series_sds in sds_by_series]
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            label=exp_label,
            color=EXPERIMENT_COLORS[exp_key],
            alpha=BAR_ALPHA,
            edgecolor="none",
            yerr=sds,
            capsize=CAPSIZE,
            ecolor=ERROR_COLOR,
            linewidth=1.0,
        )
