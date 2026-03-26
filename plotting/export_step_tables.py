#!/usr/bin/env python3
"""Export per-level step tables for Exp1-Exp4."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from common import (
    REPO_ROOT,
    filtered_participant_paths,
    human_total_level_to_model,
    is_tutorial_level,
    load_json,
    parse_participant_csv,
    should_skip_observe_level,
)


MODEL_SPECS = [
    ("FM", "full_model"),
    ("SM", "social_mentalizing"),
    ("RNM", "rational_non_mentalizing"),
    ("Non-Obs", "agent1_naive_planner"),
    ("Naive", "naive_observer"),
]
SUPPORTED_EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-level step tables for all experiments."
    )
    parser.add_argument(
        "--experiments",
        default="exp1,exp2,exp3,exp4",
        help="Comma-separated experiments to export.",
    )
    parser.add_argument(
        "--models",
        default="full_model,social_mentalizing,rational_non_mentalizing,agent1_naive_planner,naive_observer",
        help="Comma-separated models to export.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "step_tables"),
        help="Root output directory.",
    )
    return parser.parse_args()


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


def safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def validate_experiments(experiments: list[str]) -> None:
    invalid = [exp for exp in experiments if exp not in SUPPORTED_EXPERIMENTS]
    if invalid:
        raise SystemExit(
            f"Unsupported experiments: {', '.join(invalid)}. Supported: {', '.join(SUPPORTED_EXPERIMENTS)}."
        )


def resolve_models(raw_models: list[str]) -> list[tuple[str, str]]:
    valid = {model_name: label for label, model_name in MODEL_SPECS}
    resolved: list[tuple[str, str]] = []
    for model_name in raw_models:
        if model_name not in valid:
            raise SystemExit(
                f"Unsupported model: {model_name}. Supported: {', '.join(valid)}."
            )
        resolved.append((valid[model_name], model_name))
    return resolved


def step_dict_path(exp: str, model_name: str) -> Path:
    if model_name == "agent1_naive_planner":
        return REPO_ROOT / "model_outputs" / "baselines" / exp / "step_dict_naive_observer.json"
    if model_name == "full_model":
        return REPO_ROOT / "model_outputs" / "experiments" / exp / "steps_dict.json"
    return REPO_ROOT / "model_outputs" / "baselines" / exp / f"step_dict_{model_name}.json"


def reconstructed_path(exp: str, model_name: str) -> Path:
    return REPO_ROOT / "model_outputs" / "reconstructed_costs" / f"{exp}_{model_name}.json"


def human_candidates(exp: str, model_key: str) -> list[str]:
    if exp == "exp1":
        candidates = [model_key]
        if not model_key.endswith("_ascii"):
            candidates.append(f"{model_key}_ascii")
        candidates.append(f"mod_{model_key}")
        candidates.append(f"mod_{model_key}_ascii")
        if model_key.endswith("_1"):
            base = model_key[: -len("_1")]
            candidates.extend([base, f"{base}_ascii", f"mod_{base}", f"mod_{base}_ascii"])
        return candidates
    return [model_key]


def resolve_human_key(exp: str, model_key: str, human_per_case: dict) -> str | None:
    return next((candidate for candidate in human_candidates(exp, model_key) if candidate in human_per_case), None)


def load_human_total_step_medians(exp: str) -> dict[str, float]:
    human_per_case = load_json(
        REPO_ROOT / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json"
    )["per_case"]
    out: dict[str, float] = {}
    candidate_levels = set()
    for model_name in [model_name for _label, model_name in MODEL_SPECS]:
        candidate_levels.update(load_json(step_dict_path(exp, model_name)).keys())
    for level in candidate_levels:
        human_key = resolve_human_key(exp, level, human_per_case)
        if human_key is None:
            continue
        out[level] = float(human_per_case[human_key]["total_steps_median"])
    return out


def load_nonobs_total_steps(exp: str) -> dict[str, float]:
    per_case = load_json(reconstructed_path(exp, "agent1_naive_planner")).get("per_case", {})
    return {
        level: float(entry.get("total_steps", 0.0))
        for level, entry in per_case.items()
    }


def build_human_stats_single(exp: str) -> dict[str, dict[str, float]]:
    observe_values: dict[str, list[float]] = defaultdict(list)

    for csv_path in filtered_participant_paths(exp):
        per_level = parse_participant_csv(csv_path)
        for human_level, counts in per_level.items():
            if should_skip_observe_level(exp, human_level) or is_tutorial_level(human_level):
                continue
            model_level = human_total_level_to_model(exp, human_level)
            activations = float(counts.get("activation_count", 0.0))
            if activations <= 0:
                continue
            observe_values[model_level].append(float(counts.get("observe_count", 0.0)) / activations)

    stats = {}
    for level in sorted(observe_values, key=level_sort_key):
        values = observe_values[level]
        stats[level] = {
            "H_mean": safe_mean(values),
            "H_med": safe_median(values),
            "n": len(values),
        }
    return stats


def build_human_stats_multi(exp: str) -> dict[str, dict[str, float]]:
    agent2_values: dict[str, list[float]] = defaultdict(list)
    agent3_values: dict[str, list[float]] = defaultdict(list)

    for csv_path in filtered_participant_paths(exp):
        per_level = parse_participant_csv(csv_path)
        for human_level, counts in per_level.items():
            if should_skip_observe_level(exp, human_level) or is_tutorial_level(human_level):
                continue
            model_level = human_total_level_to_model(exp, human_level)
            activations = float(counts.get("activation_count", 0.0))
            if activations <= 0:
                continue
            agent2_values[model_level].append(float(counts.get("agent2_count", 0.0)) / activations)
            agent3_values[model_level].append(float(counts.get("agent3_count", 0.0)) / activations)

    stats = {}
    for level in sorted(set(agent2_values) | set(agent3_values), key=level_sort_key):
        a2 = agent2_values.get(level, [])
        a3 = agent3_values.get(level, [])
        stats[level] = {
            "H2_mean": safe_mean(a2),
            "H2_med": safe_median(a2),
            "H3_mean": safe_mean(a3),
            "H3_med": safe_median(a3),
            "n": max(len(a2), len(a3)),
        }
    return stats


def build_single_table(exp: str, model_label: str, model_name: str) -> str:
    human = build_human_stats_single(exp)
    step_dict = load_json(step_dict_path(exp, model_name))
    human_total_medians = load_human_total_step_medians(exp)
    nonobs_total_steps = load_nonobs_total_steps(exp)

    rows = []
    for level in sorted(step_dict.keys(), key=level_sort_key):
        if is_tutorial_level(level) or level not in human:
            continue
        step_value = step_dict[level]
        step_total = float(step_value if not isinstance(step_value, dict) else step_value.get("t", 0.0))
        if model_name == "agent1_naive_planner":
            step_total = 0.0
        h = human[level]
        rows.append(
            {
                "Level": level,
                "H_mean": h["H_mean"],
                "H_med": h["H_med"],
                "Model": step_total,
                "Step_T": step_total,
                "H_step_med": human_total_medians.get(level, 0.0),
                "NonObs_total": nonobs_total_steps.get(level, 0.0),
                "n": int(h["n"]),
            }
        )

    headers = ["Level", "H_mean", "H_med", "Model", "Step_T", "H_step_med", "NonObs_total", "n"]
    return render_table(
        exp=exp,
        model_label=model_label,
        model_name=model_name,
        headers=headers,
        rows=rows,
        signed_headers=set(),
    )


def build_multi_table(exp: str, model_label: str, model_name: str) -> str:
    human = build_human_stats_multi(exp)
    step_dict = load_json(step_dict_path(exp, model_name))
    human_total_medians = load_human_total_step_medians(exp)
    nonobs_total_steps = load_nonobs_total_steps(exp)

    rows = []
    for level in sorted(step_dict.keys(), key=level_sort_key):
        if is_tutorial_level(level) or level not in human:
            continue
        step_entry = step_dict.get(level, {})
        agent2_count = float(step_entry.get("agent2_count", 0.0))
        agent3_count = float(step_entry.get("agent3_count", 0.0))
        step_total = float(step_entry.get("t", agent2_count + agent3_count))
        if model_name == "agent1_naive_planner":
            agent2_count = 0.0
            agent3_count = 0.0
            step_total = 0.0
        h = human[level]
        rows.append(
            {
                "Level": level,
                "H2_mean": h["H2_mean"],
                "H2_med": h["H2_med"],
                "M2": agent2_count,
                "H3_mean": h["H3_mean"],
                "H3_med": h["H3_med"],
                "M3": agent3_count,
                "Step_T": step_total,
                "H_step_med": human_total_medians.get(level, 0.0),
                "NonObs_total": nonobs_total_steps.get(level, 0.0),
                "n": int(h["n"]),
            }
        )

    headers = [
        "Level",
        "H2_mean",
        "H2_med",
        "M2",
        "H3_mean",
        "H3_med",
        "M3",
        "Step_T",
        "H_step_med",
        "NonObs_total",
        "n",
    ]
    return render_table(
        exp=exp,
        model_label=model_label,
        model_name=model_name,
        headers=headers,
        rows=rows,
        signed_headers=set(),
    )


def render_table(
    *,
    exp: str,
    model_label: str,
    model_name: str,
    headers: list[str],
    rows: list[dict[str, float | int | str]],
    signed_headers: set[str],
) -> str:
    widths: dict[str, int] = {}
    for header in headers:
        values = [header]
        for row in rows:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:+.1f}" if header in signed_headers else f"{value:.1f}")
            else:
                values.append(str(value))
        widths[header] = max(len(v) for v in values)

    lines = [
        f"Experiment: {exp}",
        f"Model: {model_label} ({model_name})",
        "",
        "  ".join(
            header.ljust(widths[header]) if header == "Level" else header.rjust(widths[header])
            for header in headers
        ),
        "  ".join("-" * widths[header] for header in headers),
    ]

    for row in rows:
        fields = []
        for header in headers:
            value = row[header]
            if header == "Level":
                fields.append(str(value).ljust(widths[header]))
            elif isinstance(value, float):
                if header in signed_headers:
                    fields.append(f"{value:+.1f}".rjust(widths[header]))
                else:
                    fields.append(f"{value:.1f}".rjust(widths[header]))
            else:
                fields.append(str(value).rjust(widths[header]))
        lines.append("  ".join(fields))
    return "\n".join(lines)


def build_table(exp: str, model_label: str, model_name: str) -> str:
    if exp in {"exp1", "exp2"}:
        return build_single_table(exp, model_label, model_name)
    return build_multi_table(exp, model_label, model_name)


def main() -> None:
    args = parse_args()
    experiments = parse_csv_arg(args.experiments)
    validate_experiments(experiments)
    models = resolve_models(parse_csv_arg(args.models))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        exp_dir = output_root / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        for model_label, model_name in models:
            table = build_table(exp, model_label, model_name)
            output_stem = "non_obs" if model_name == "agent1_naive_planner" else model_name
            out_path = exp_dir / f"{output_stem}_step_table.txt"
            out_path.write_text(table + "\n", encoding="utf-8")
            print(f"Saved table -> {out_path}")


if __name__ == "__main__":
    main()
