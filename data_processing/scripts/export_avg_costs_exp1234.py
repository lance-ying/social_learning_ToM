#!/usr/bin/env python3
"""
Export average cost summaries for experiments 1-4 into one compact JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4")
MODEL_LABELS = (
    "full_model",
    "social_mentalizing",
    "rational_non_mentalizing",
    "naive_observer",
)
METRICS = (
    "move_cost",
    "observe_cost",
    "interaction_cost",
    "planning_cost",
    "total_cost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export average human/model costs for experiments 1-4 into one JSON file."
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="total_cost",
        help="Cost metric to export from the JSON summary blocks.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional output path. Defaults to data_processing/outputs/avg_<metric>s_exp1234.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_costs_path(repo_root: Path, exp: str) -> Path:
    if exp == "exp4":
        override = os.environ.get("EXP4_HUMAN_COSTS_FILE", "").strip()
        if override:
            return Path(override)
    return repo_root / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json"


def model_dir(repo_root: Path) -> Path:
    default_dir = repo_root / "model_outputs" / "reconstructed_costs"
    if default_dir.exists():
        return default_dir
    return repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"


def summary_mean_key(metric: str) -> str:
    return f"mean_{metric}"


def summary_median_key(metric: str) -> str:
    return f"median_{metric}"


def extract_summary_value(path: Path, key: str) -> float:
    data = load_json(path)
    summary = data.get("summary", {})
    if key not in summary:
        raise KeyError(f"Missing summary key {key!r} in {path}")
    return float(summary[key])


def build_output(repo_root: Path, metric: str) -> dict:
    out = {"metric": summary_mean_key(metric)}
    models_root = model_dir(repo_root)
    mean_key = summary_mean_key(metric)
    median_key = summary_median_key(metric)

    for exp in EXPERIMENTS:
        human_path = human_costs_path(repo_root, exp)
        exp_out = {
            "human_mean": extract_summary_value(human_path, mean_key),
            "human_median": extract_summary_value(human_path, median_key),
        }
        for label in MODEL_LABELS:
            exp_out[label] = extract_summary_value(models_root / f"{exp}_{label}.json", mean_key)
        out[exp] = exp_out

    return out


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root / "data_processing" / "outputs" / f"avg_{args.metric}s_exp1234.json"
    )

    output = build_output(repo_root, args.metric)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Saved -> {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
