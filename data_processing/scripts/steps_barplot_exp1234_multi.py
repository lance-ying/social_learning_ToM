#!/usr/bin/env python3
"""
Grouped total-steps barplot across experiments 1-4.

Steps include observations. Model categories are on the x-axis, with
experiments grouped within each category.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grouped_barplot_style import (
    EXPERIMENTS,
    SERIES,
    FIGSIZE_SINGLE,
    LEGEND_FONTSIZE,
    apply_axis_style,
    plot_grouped_bars,
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_exp1_model_key(model_key: str) -> str:
    if model_key.startswith("mod_") and model_key.endswith("_ascii"):
        return model_key.removeprefix("mod_").removesuffix("_ascii")
    return model_key.split("_")[0]


def exp1_human_candidates(label: str, model_key: str) -> list[str]:
    normalized = normalize_exp1_model_key(model_key)
    return [f"mod_{normalized}_ascii"]


def default_human_candidates(_label: str, model_key: str) -> list[str]:
    return [model_key]


def human_candidates(exp: str, label: str, model_key: str) -> list[str]:
    if exp == "exp1":
        return exp1_human_candidates(label, model_key)
    return default_human_candidates(label, model_key)


def common_matched_keys(repo_root: Path, exp: str) -> tuple[dict[str, dict[str, float]], dict]:
    model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"
    human_dir = repo_root / "data_processing" / "outputs" / "human_costs"
    human_per_case = load_json(human_dir / f"{exp}_human_costs.json")["per_case"]

    series_data = {}
    matched_human_keys = []

    for _series_label, key in SERIES[1:]:
        model_per_case = load_json(model_dir / f"{exp}_{key}.json")["per_case"]
        aligned = {}
        for model_key, model_value in model_per_case.items():
            human_key = next(
                (candidate for candidate in human_candidates(exp, key, model_key) if candidate in human_per_case),
                None,
            )
            if human_key is not None:
                aligned[human_key] = model_value
        series_data[key] = aligned
        matched_human_keys.append(set(aligned))

    common_keys = set(human_per_case)
    for key_set in matched_human_keys:
        common_keys &= key_set

    common_keys = sorted(common_keys)
    filtered_series = {
        key: {human_key: value for human_key, value in aligned.items() if human_key in common_keys}
        for key, aligned in series_data.items()
    }
    return filtered_series, {key: human_per_case[key] for key in common_keys}


def human_mean_sd_total_steps(per_case: dict) -> tuple[float, float]:
    values = [float(v["total_steps_mean"]) for v in per_case.values() if "total_steps_mean" in v]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def model_mean_sd_total_steps(per_case: dict) -> tuple[float, float]:
    values = [float(v["total_steps"]) for v in per_case.values() if "total_steps" in v]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    output_file = repo_root / "data_processing" / "outputs" / "plots" / "steps_barplot_exp1234_multi.png"
    matched_by_experiment = {exp: common_matched_keys(repo_root, exp) for exp in EXPERIMENTS}

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    means_by_series = []
    sds_by_series = []

    for _series_label, series_key in SERIES:
        means = []
        sds = []

        for exp in EXPERIMENTS:
            series_data, human_common = matched_by_experiment[exp]
            if series_key == "human":
                mean_val, sd_val = human_mean_sd_total_steps(human_common)
            else:
                mean_val, sd_val = model_mean_sd_total_steps(series_data[series_key])
            means.append(mean_val)
            sds.append(sd_val)

        means_by_series.append(means)
        sds_by_series.append(sds)

    plot_grouped_bars(ax, means_by_series, sds_by_series)
    apply_axis_style(ax, "Mean Total Steps", "Total Steps (Including Observations)")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
