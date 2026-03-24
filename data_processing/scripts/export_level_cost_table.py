#!/usr/bin/env python3
"""Export a per-level total-cost table for one experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_COLUMNS = [
    ("FM", "full_model"),
    ("SM", "social_mentalizing"),
    ("RNM", "rational_non_mentalizing"),
    ("Naive", "naive_observer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a per-level total-cost comparison table."
    )
    parser.add_argument(
        "--exp",
        required=True,
        choices=["exp1", "exp2", "exp3", "exp4"],
        help="Experiment to export.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to save the table text.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def human_candidates(exp: str, model_key: str) -> list[str]:
    if exp == "exp1":
        candidates = [model_key]
        if model_key.endswith("_1"):
            candidates.append(model_key[: -len("_1")])
        if not model_key.endswith("_ascii"):
            candidates.extend([f"{candidate}_ascii" for candidate in list(candidates)])
        return candidates
    return [model_key]


def resolve_human_key(exp: str, model_key: str, human_per_case: dict) -> str | None:
    return next((candidate for candidate in human_candidates(exp, model_key) if candidate in human_per_case), None)


def level_sort_key(level: str):
    if "_scenario" in level:
        prefix, suffix = level.split("_scenario", 1)
        try:
            return (prefix, int(suffix))
        except ValueError:
            return (prefix, suffix)
    if "_" in level:
        prefix, suffix = level.rsplit("_", 1)
        try:
            return (prefix, int(suffix))
        except ValueError:
            return (prefix, suffix)
    return (level, 0)


def build_table(exp: str, repo_root: Path) -> str:
    human_per_case = load_json(
        repo_root / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json"
    )["per_case"]
    model_data = {
        label: load_json(
            repo_root / "model_outputs" / "reconstructed_costs" / f"{exp}_{model_name}.json"
        )["per_case"]
        for label, model_name in MODEL_COLUMNS
    }

    common_keys = set()
    for label, per_case in model_data.items():
        for model_key in per_case:
            human_key = resolve_human_key(exp, model_key, human_per_case)
            if human_key is not None:
                common_keys.add((model_key, human_key))

    rows = []
    for model_key, human_key in sorted(common_keys, key=lambda item: level_sort_key(item[0])):
        row = {
            "Level": model_key,
            "H_mean": float(human_per_case[human_key]["total_cost_mean"]),
            "H_med": float(human_per_case[human_key]["total_cost_median"]),
            "n": int(human_per_case[human_key]["n_participants"]),
        }
        include = True
        for label, _model_name in MODEL_COLUMNS:
            per_case = model_data[label]
            if model_key not in per_case or "total_cost" not in per_case[model_key]:
                include = False
                break
            row[label] = float(per_case[model_key]["total_cost"])
        if include:
            rows.append(row)

    headers = ["Level", "H_mean", "H_med"] + [label for label, _ in MODEL_COLUMNS] + ["n"]
    widths = {}
    for header in headers:
        values = [header]
        for row in rows:
            val = row[header]
            values.append(f"{val:.1f}" if isinstance(val, float) else str(val))
        widths[header] = max(len(v) for v in values)

    lines = []
    lines.append(
        "  ".join(
            header.ljust(widths[header]) if header == "Level" else header.rjust(widths[header])
            for header in headers
        )
    )
    lines.append(
        "  ".join(
            ("-" * widths[header]) for header in headers
        )
    )
    for row in rows:
        fields = []
        for header in headers:
            val = row[header]
            if header == "Level":
                fields.append(str(val).ljust(widths[header]))
            elif isinstance(val, float):
                fields.append(f"{val:.1f}".rjust(widths[header]))
            else:
                fields.append(str(val).rjust(widths[header]))
        lines.append("  ".join(fields))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    table = build_table(args.exp, repo_root)
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n")
        print(f"Saved table -> {out_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()
