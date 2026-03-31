from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = REPO_ROOT / "data_processing" / "data_processed"
PLOTS_DIR = REPO_ROOT / "plotting" / "outputs"
SCRIPTS_DIR = REPO_ROOT / "data_processing" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def apply_reference_style(*args, **kwargs):
    from correlation_4panel_style import apply_reference_style as _apply_reference_style

    return _apply_reference_style(*args, **kwargs)


def bootstrap_ccc_ci(*args, **kwargs):
    from correlation_4panel_style import bootstrap_ccc_ci as _bootstrap_ccc_ci

    return _bootstrap_ccc_ci(*args, **kwargs)


def bootstrap_r_ci(*args, **kwargs):
    from correlation_4panel_style import bootstrap_r_ci as _bootstrap_r_ci

    return _bootstrap_r_ci(*args, **kwargs)


def plot_points_errorbars_and_fit(*args, **kwargs):
    from correlation_4panel_style import plot_points_errorbars_and_fit as _plot_points_errorbars_and_fit

    return _plot_points_errorbars_and_fit(*args, **kwargs)


EXPERIMENTS = ["exp1", "exp2", "exp3", "exp4"]
EXPERIMENT_LABELS = {
    "exp1": "Experiment 1",
    "exp2": "Experiment 2",
    "exp3": "Experiment 3",
    "exp4": "Experiment 4",
}
EXPERIMENT_COLORS = {
    "exp1": "#444444",
    "exp2": "#c95f5f",
    "exp3": "#d98e3d",
    "exp4": "#5b8fd1",
}
MODEL_PANELS = [
    ("Rational Mentalizing\n(Full Model)", "full_model"),
    ("Social Mentalizing", "social_mentalizing"),
    ("Rational Non-Mentalizing", "rational_non_mentalizing"),
    ("Naive Planner", "agent1_naive_planner"),
    ("Naive Observer", "naive_observer"),
]
OBSERVE_MODEL_PANELS = [panel for panel in MODEL_PANELS if panel[1] != "agent1_naive_planner"]
EXTRA_MODEL_PANELS = [
    ("Social Mentalizing\nUntil One Converges", "social_mentalizing_until_one_converges"),
    ("RNM Expert Only\nUntil Expert Wizard", "rational_non_mentalizing_expert_only_until_expert_wizard"),
    (
        "RNM Novice Full + Expert\nUntil Expert Wizard",
        "rational_non_mentalizing_novice_full_expert_until_expert_wizard",
    ),
    ("Naive Expert Only\nUntil Expert Wizard", "naive_observer_expert_only_until_expert_wizard"),
    (
        "Naive Novice Full + Expert\nUntil Expert Wizard",
        "naive_observer_novice_full_expert_until_expert_wizard",
    ),
]
ALL_MODEL_PANELS = MODEL_PANELS + EXTRA_MODEL_PANELS
ALL_OBSERVE_MODEL_PANELS = [panel for panel in ALL_MODEL_PANELS if panel[1] != "agent1_naive_planner"]
EXP34_ONLY_MODELS = set()
EXP4_ONLY_MODELS = set()
NON_TASK_LEVELS = {"comprehension_check", "experiment"}
EXP34_OBSERVE_SKIP_LEVELS = {"s111_1"}
EXP34_OBSERVE_SKIP_PREFIXES = ("sm111_", "sm112_")
TUTORIAL_LEVEL_PREFIXES = ("s111", "s112", "sm111", "sm112", "mod_s111", "mod_s112")
LEVEL_RE = re.compile(r"^Level:\s*(.+?)\s*$", re.IGNORECASE)
EXP34_LEVEL_RE = re.compile(r"^(sm\d+)_(\d+)$")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_panels(model_names: list[str] | None = None, *, observe_only: bool = False) -> list[tuple[str, str]]:
    default_panels = OBSERVE_MODEL_PANELS if observe_only else MODEL_PANELS
    supported_panels = ALL_OBSERVE_MODEL_PANELS if observe_only else ALL_MODEL_PANELS
    if model_names is None:
        return list(default_panels)

    panel_by_model = {model_name: label for label, model_name in supported_panels}
    resolved = []
    for model_name in model_names:
        if model_name not in panel_by_model:
            supported = ", ".join(name for _label, name in supported_panels)
            raise ValueError(f"Unsupported model: {model_name}. Supported: {supported}.")
        resolved.append((panel_by_model[model_name], model_name))
    return resolved


def model_supported_in_exp(exp: str, model_name: str) -> bool:
    if model_name in EXP4_ONLY_MODELS:
        return exp == "exp4"
    if model_name in EXP34_ONLY_MODELS:
        return exp in {"exp3", "exp4"}
    return True


def effective_model_name(exp: str, model_name: str) -> str:
    if model_name == "social_mentalizing_until_one_converges" and exp in {"exp1", "exp2"}:
        return "social_mentalizing"
    if model_name in {
        "rational_non_mentalizing_expert_only_until_expert_wizard",
        "rational_non_mentalizing_novice_full_expert_until_expert_wizard",
    } and exp in {"exp1", "exp2", "exp3"}:
        return "rational_non_mentalizing"
    if model_name in {
        "naive_observer_expert_only_until_expert_wizard",
        "naive_observer_novice_full_expert_until_expert_wizard",
    } and exp in {"exp1", "exp2", "exp3"}:
        return "naive_observer"
    return model_name


def preferred_model_name(exp: str, model_name: str) -> str:
    if model_name == "social_mentalizing" and exp in {"exp3", "exp4"}:
        return "social_mentalizing_until_one_converges"
    if model_name == "social_mentalizing_until_one_converges" and exp in {"exp1", "exp2"}:
        return "social_mentalizing"
    return model_name


def make_output_path(filename: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR / filename


def pretty_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def sample_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def human_level_to_model(exp: str, level: str) -> str:
    if exp == "exp1" and level.startswith("mod_"):
        return level.removeprefix("mod_")
    if exp not in {"exp3", "exp4"}:
        return level
    match = EXP34_LEVEL_RE.match(level)
    if match:
        return f"{match.group(1)}_scenario{match.group(2)}"
    return level


def human_total_level_to_model(exp: str, level: str) -> str:
    if exp == "exp1":
        normalized = level.removeprefix("mod_")
        if normalized.endswith("_1"):
            return normalized[:-2]
        return normalized
    return human_level_to_model(exp, level)


def should_skip_observe_level(exp: str, level: str) -> bool:
    if level in NON_TASK_LEVELS:
        return True
    if exp in {"exp3", "exp4"}:
        return level in EXP34_OBSERVE_SKIP_LEVELS or level.startswith(EXP34_OBSERVE_SKIP_PREFIXES)
    return False


def is_tutorial_level(level: str) -> bool:
    return level.startswith(TUTORIAL_LEVEL_PREFIXES)


@lru_cache(maxsize=None)
def parse_participant_csv(path: Path) -> dict[str, dict[str, float]]:
    per_level: dict[str, dict[str, float]] = {}
    current_level: str | None = None
    header_map: dict[str, int] | None = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                header_map = None
                continue

            match = LEVEL_RE.match(line)
            if match:
                current_level = match.group(1)
                header_map = None
                continue

            if current_level is None:
                continue

            row = next(csv.reader([raw_line]))
            if not row:
                continue

            if header_map is None:
                header_map = {col.strip().lower(): idx for idx, col in enumerate(row)}
                per_level.setdefault(
                    current_level,
                    {
                        "total_steps": 0.0,
                        "observe_count": 0.0,
                        "agent2_count": 0.0,
                        "agent3_count": 0.0,
                        "activation_count": 0.0,
                        "steps_remaining": np.nan,
                    },
                )
                per_level[current_level]["activation_count"] += 1.0
                continue

            type_idx = header_map.get("type")
            if type_idx is None or len(row) <= type_idx:
                continue

            event_type = row[type_idx].strip().upper()
            if not event_type:
                continue

            stats = per_level.setdefault(
                current_level,
                {
                    "total_steps": 0.0,
                    "observe_count": 0.0,
                    "agent2_count": 0.0,
                    "agent3_count": 0.0,
                    "activation_count": 0.0,
                    "steps_remaining": np.nan,
                },
            )

            if event_type == "GAME_ACTION":
                stats["total_steps"] += 1.0
                continue

            if event_type == "AGENT_OBSERVE":
                stats["total_steps"] += 1.0
                stats["observe_count"] += 1.0
                observed_agent_idx = header_map.get("observed agent id")
                if observed_agent_idx is not None and len(row) > observed_agent_idx:
                    observed_agent = row[observed_agent_idx].strip()
                    if observed_agent in {"2", "2.0"}:
                        stats["agent2_count"] += 1.0
                    elif observed_agent in {"3", "3.0"}:
                        stats["agent3_count"] += 1.0
                continue

            if event_type == "INTERACTION" and current_level not in NON_TASK_LEVELS:
                interaction_type_idx = header_map.get("interaction type")
                interaction_type = ""
                if interaction_type_idx is not None and len(row) > interaction_type_idx:
                    interaction_type = row[interaction_type_idx].strip().lower()
                if interaction_type != "comprehension_check":
                    stats["total_steps"] += 1.0
                continue

            if event_type == "LEVEL_COMPLETE":
                steps_remaining_idx = header_map.get("steps remaining")
                if steps_remaining_idx is not None and len(row) > steps_remaining_idx:
                    raw_value = row[steps_remaining_idx].strip()
                    if raw_value:
                        try:
                            stats["steps_remaining"] = float(raw_value)
                        except ValueError:
                            pass

    return per_level


@lru_cache(maxsize=None)
def participant_quality_scores(exp: str) -> dict[Path, float]:
    scores: dict[Path, float] = {}
    csv_dir = DATA_PROCESSED_DIR / exp
    for path in sorted(csv_dir.glob("*.csv")):
        total_score = 0.0
        for level, counts in parse_participant_csv(path).items():
            if level in NON_TASK_LEVELS or is_tutorial_level(level):
                continue
            steps_remaining = counts.get("steps_remaining", np.nan)
            if np.isnan(steps_remaining):
                continue
            total_score += float(steps_remaining)
        scores[path] = total_score
    return scores


@lru_cache(maxsize=None)
def participant_quality_threshold(exp: str) -> float:
    scores = list(participant_quality_scores(exp).values())
    if not scores:
        return float("-inf")
    q1 = float(np.percentile(scores, 25))
    q3 = float(np.percentile(scores, 75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr


@lru_cache(maxsize=None)
def filtered_participant_paths(exp: str) -> tuple[Path, ...]:
    scores = participant_quality_scores(exp)
    threshold = participant_quality_threshold(exp)
    return tuple(sorted(path for path, score in scores.items() if score >= threshold))


@lru_cache(maxsize=None)
def aggregate_human_total_steps(exp: str) -> dict[str, dict[str, float]]:
    per_level_values: dict[str, list[float]] = defaultdict(list)
    for path in filtered_participant_paths(exp):
        for level, counts in parse_participant_csv(path).items():
            if level in NON_TASK_LEVELS:
                continue
            model_level = human_total_level_to_model(exp, level)
            per_level_values[model_level].append(float(counts["total_steps"]))

    return {
        level: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": sample_sd(values),
            "n": len(values),
        }
        for level, values in per_level_values.items()
    }


@lru_cache(maxsize=None)
def aggregate_human_total_cost(exp: str) -> dict[str, dict[str, float]]:
    raw = load_json(REPO_ROOT / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json")
    per_case = raw.get("per_case", {})
    return {
        human_total_level_to_model(exp, level): {
            "mean": float(stats["total_cost_mean"]),
            "median": float(stats["total_cost_median"]),
            "sd": float(stats["total_cost_sd"]),
            "n": int(stats["n_participants"]),
        }
        for level, stats in per_case.items()
    }


@lru_cache(maxsize=None)
def aggregate_human_observes(exp: str) -> dict[str, dict[str, float]]:
    combined_values: dict[str, list[float]] = defaultdict(list)
    agent2_values: dict[str, list[float]] = defaultdict(list)
    agent3_values: dict[str, list[float]] = defaultdict(list)

    for path in filtered_participant_paths(exp):
        for level, counts in parse_participant_csv(path).items():
            if should_skip_observe_level(exp, level):
                continue
            model_level = human_level_to_model(exp, level)
            activation_count = float(counts.get("activation_count", 0.0))
            if activation_count <= 0:
                continue
            combined_values[model_level].append(float(counts["observe_count"]) / activation_count)
            agent2_values[model_level].append(float(counts["agent2_count"]) / activation_count)
            agent3_values[model_level].append(float(counts["agent3_count"]) / activation_count)

    all_levels = set(combined_values) | set(agent2_values) | set(agent3_values)
    return {
        level: {
            "combined_mean": float(np.mean(combined_values.get(level, [0.0]))),
            "combined_sd": sample_sd(combined_values.get(level, [])),
            "agent2_mean": float(np.mean(agent2_values.get(level, [0.0]))),
            "agent2_sd": sample_sd(agent2_values.get(level, [])),
            "agent3_mean": float(np.mean(agent3_values.get(level, [0.0]))),
            "agent3_sd": sample_sd(agent3_values.get(level, [])),
        }
        for level in all_levels
    }


def load_model_observe_predictions(exp: str, model_name: str) -> dict:
    if not model_supported_in_exp(exp, model_name):
        return {}
    model_name = effective_model_name(exp, model_name)
    if model_name == "full_model":
        path = REPO_ROOT / "model_outputs" / "experiments" / exp / "steps_dict.json"
        return load_json(path) if path.exists() else {}
    if model_name == "agent1_naive_planner":
        path = REPO_ROOT / "model_outputs" / "reconstructed_costs" / f"{exp}_{model_name}.json"
        return load_json(path)["per_case"] if path.exists() else {}
    path = REPO_ROOT / "model_outputs" / "baselines" / exp / f"step_dict_{model_name}.json"
    return load_json(path) if path.exists() else {}


def load_model_total_steps_predictions(exp: str, model_name: str) -> dict[str, dict]:
    if not model_supported_in_exp(exp, model_name):
        return {}
    model_name = effective_model_name(exp, model_name)
    path = REPO_ROOT / "model_outputs" / "reconstructed_costs" / f"{exp}_{model_name}.json"
    if not path.exists():
        return {}
    per_case = load_json(path)["per_case"]
    if exp == "exp1":
        return {
            (level[:-2] if level.endswith("_1") else level): value
            for level, value in per_case.items()
        }
    return per_case


def observe_model_scalar(exp: str, model_value, observe_metric: str) -> float:
    if isinstance(model_value, (int, float)):
        return float(model_value)

    if "observe_steps" in model_value:
        if observe_metric == "combined":
            return float(model_value["observe_steps"])

        events = model_value.get("observation_events", [])
        if observe_metric == "agent2":
            return float(sum(1 for event in events if event.get("observed_agent") == "agent2"))
        if observe_metric == "agent3":
            return float(sum(1 for event in events if event.get("observed_agent") == "agent3"))
        return float(model_value["observe_steps"])

    if exp in {"exp1", "exp2"}:
        if "t" in model_value:
            return float(model_value["t"])
        return float(model_value.get("observations_count", 0.0))

    if observe_metric == "agent2":
        return float(model_value.get("agent2_count", 0.0))
    if observe_metric == "agent3":
        return float(model_value.get("agent3_count", 0.0))
    if "t" in model_value:
        return float(model_value["t"])
    return float(model_value.get("agent2_count", 0.0) + model_value.get("agent3_count", 0.0))


def collect_observe_pairs(
    exp: str, model_name: str, observe_metric: str = "combined"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model_predictions = load_model_observe_predictions(exp, model_name)
    human_stats = aggregate_human_observes(exp)
    human_mean_key = f"{observe_metric}_mean"
    human_sd_key = f"{observe_metric}_sd"

    model_vals = []
    human_means = []
    human_sds = []
    keys = []

    for level, model_value in model_predictions.items():
        if level not in human_stats:
            continue
        model_vals.append(observe_model_scalar(exp, model_value, observe_metric))
        human_means.append(float(human_stats[level][human_mean_key]))
        human_sds.append(float(human_stats[level][human_sd_key]))
        keys.append(level)

    return (
        np.asarray(model_vals, dtype=float),
        np.asarray(human_means, dtype=float),
        np.asarray(human_sds, dtype=float),
        keys,
    )


def collect_total_steps_pairs(exp: str, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model_predictions = load_model_total_steps_predictions(exp, model_name)
    human_stats = aggregate_human_total_steps(exp)

    model_vals = []
    human_means = []
    human_sds = []
    keys = []

    for level, model_value in model_predictions.items():
        if level not in human_stats or "total_steps" not in model_value:
            continue
        model_vals.append(float(model_value["total_steps"]))
        human_means.append(float(human_stats[level]["mean"]))
        human_sds.append(float(human_stats[level]["sd"]))
        keys.append(level)

    return (
        np.asarray(model_vals, dtype=float),
        np.asarray(human_means, dtype=float),
        np.asarray(human_sds, dtype=float),
        keys,
    )


def collect_total_cost_pairs(exp: str, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model_predictions = load_model_total_steps_predictions(exp, model_name)
    human_stats = aggregate_human_total_cost(exp)

    model_vals = []
    human_means = []
    human_sds = []
    keys = []

    for level, model_value in model_predictions.items():
        if level not in human_stats or "total_cost" not in model_value:
            continue
        model_vals.append(float(model_value["total_cost"]))
        human_means.append(float(human_stats[level]["mean"]))
        human_sds.append(float(human_stats[level]["sd"]))
        keys.append(level)

    return (
        np.asarray(model_vals, dtype=float),
        np.asarray(human_means, dtype=float),
        np.asarray(human_sds, dtype=float),
        keys,
    )


def common_total_step_keys(exp: str) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    human_stats = aggregate_human_total_steps(exp)
    model_predictions = {
        model_name: load_model_total_steps_predictions(exp, model_name)
        for _label, model_name in ALL_MODEL_PANELS
        if model_supported_in_exp(exp, model_name)
    }

    common_keys = set(human_stats)
    for model_name, model_dict in model_predictions.items():
        if not model_dict:
            continue
        common_keys &= {level for level, value in model_dict.items() if "total_steps" in value}

    common_keys = sorted(common_keys)
    filtered_models = {
        model_name: {level: model_dict[level] for level in common_keys}
        for model_name, model_dict in model_predictions.items()
    }
    filtered_humans = {level: human_stats[level] for level in common_keys}
    return filtered_models, filtered_humans


def annotate_stats(ax, x: np.ndarray, y: np.ndarray, fontsize: int = 18) -> None:
    if len(x) < 3:
        return
    r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
    ccc, _ccc_low, _ccc_high = bootstrap_ccc_ci(x, y, n_resamples=1000)
    ax.text(
        0.05,
        0.90,
        f"r = {r:.2f}\nCI = [{ci_low:.2f}, {ci_high:.2f}]\nCCC = {ccc:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#1a1a1a",
    )


def pooled_limits(panel_data: list[list[tuple[str, np.ndarray, np.ndarray]]]) -> tuple[float, float]:
    all_values: list[float] = []
    for by_exp in panel_data:
        for _exp, x, y in by_exp:
            all_values.extend(x.tolist())
            all_values.extend(y.tolist())
    if not all_values:
        return 0.0, 1.0
    data_min = float(min(all_values))
    data_max = float(max(all_values))
    pad = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    return data_min - pad, data_max + pad
