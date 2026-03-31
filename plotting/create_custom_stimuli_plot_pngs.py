#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import filtered_participant_paths, parse_participant_csv

BG = "#ffffff"
BOX = "#707070"
SERIES_A1 = "#de6b8a"
SERIES_A2 = "#f0b35a"
EXP2_MUSTARD_S1 = "#e79a2f"
EXP2_MUSTARD_S2 = "#f2b24d"
AGENT2_BLUE_S1 = "#2f6df3"
AGENT3_GREEN_S1 = "#22a15a"
LABEL_TEXT = "#111111"
PAIR_BLUE_DARK = "#2f6df3"
PAIR_BLUE_LIGHT = "#8fb7ff"
EXP12_SPLIT_BLUE_SHADES = ["#1f5fe0", "#3d78ee", "#5d91f4", "#7baaf8", "#9cc2fb"]
EXP12_SPLIT_HATCHES = ["", "//", "..", "xx", "\\\\"]
EXP12_SPLIT_LABELS = ["Human", "Rat. Ment.", "Soc. Ment.", "Rat. Non-M.", "Naive"]
AGENT2_BLUE_SHADES = ["#1f5fe0", "#3d78ee", "#5d91f4", "#7baaf8", "#9cc2fb"]
AGENT3_GREEN_SHADES = ["#13824a", "#23985a", "#3aae6f", "#63c78f", "#91ddb2"]
MODEL_HATCHES = ["", "//", "..", "xx", "\\\\"]

GROUP_KEYS = ["human", "full", "social", "nonmental", "naive"]
GROUP_LABELS = [
    "Human",
    "Rational\nMentalizing",
    "Social\nMentalizing",
    "Rational\nNon-\nmentalizing",
    "Naive",
]
ABBREV_GROUP_LABELS = ["H", "RM", "SM", "RN", "N"]
LEVEL_RE = re.compile(r"^Level:\s*(.+?)\s*$", re.IGNORECASE)
SKIP_LEVELS = {"comprehension_check", "experiment", "s111_1"}
SKIP_PREFIXES = ("sm111_", "sm112_")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_results_py(path: Path) -> dict:
    ns: dict = {}
    exec(path.read_text(encoding="utf-8"), ns)
    return ns.get("results_dict", {})


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _map_level_name(human_level: str) -> str:
    m = re.match(r"^(sm\d+)_(\d+)$", human_level)
    if m:
        return f"{m.group(1)}_scenario{m.group(2)}"

    m = re.match(r"^(sm\d+)_true$", human_level)
    if m:
        return f"{m.group(1)}_scenario1"

    return human_level


def _parse_single_agent_csv(path: Path) -> dict:
    level_data = defaultdict(lambda: {"observe_count": 0, "activation_count": 0})
    current_level = None
    in_table = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue

            m = LEVEL_RE.match(line)
            if m:
                current_level = m.group(1)
                in_table = False
                continue

            if current_level and line.lower().startswith("timestamp,type"):
                in_table = True
                level_data[current_level]["activation_count"] += 1
                continue

            if current_level and in_table:
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = line.split(",")

                if len(row) >= 2:
                    type_field = (row[1] or "").strip().upper()
                    if "OBSERVE" in type_field:
                        level_data[current_level]["observe_count"] += 1

    return dict(level_data)


def _parse_multi_agent_csv(path: Path) -> dict:
    level_data = defaultdict(
        lambda: {"agent2_count": 0, "agent3_count": 0, "activation_count": 0}
    )
    current_level = None
    in_table = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue

            m = LEVEL_RE.match(line)
            if m:
                current_level = m.group(1)
                in_table = False
                continue

            if current_level and line.lower().startswith("timestamp,type"):
                in_table = True
                level_data[current_level]["activation_count"] += 1
                continue

            if current_level and in_table:
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = line.split(",")

                if len(row) >= 6:
                    type_field = (row[1] or "").strip().upper()
                    if "OBSERVE" not in type_field:
                        continue
                    agent_id = (row[5] or "").strip()
                    if agent_id in ("2", "2.0"):
                        level_data[current_level]["agent2_count"] += 1
                    elif agent_id in ("3", "3.0"):
                        level_data[current_level]["agent3_count"] += 1

    return dict(level_data)


def _build_human_stats_exp12(csv_dir: Path) -> dict:
    per_level_means = defaultdict(list)

    exp = csv_dir.name
    for csv_path in filtered_participant_paths(exp):
        file_data = parse_participant_csv(csv_path)
        for level, counts in file_data.items():
            if level in SKIP_LEVELS or any(level.startswith(pfx) for pfx in SKIP_PREFIXES):
                continue
            activations = float(counts.get("activation_count", 0.0))
            if activations <= 0:
                continue
            per_level_means[level].append(float(counts.get("observe_count", 0.0)) / activations)

    out = {}
    for level, values in per_level_means.items():
        out[level] = {
            "mean_observe_per_activation": _safe_mean(values),
            "raw_sem": _safe_sem(values),
            "samples": list(values),
        }
    return out


def _build_human_stats_exp34(csv_dir: Path) -> dict:
    per_level_agent2 = defaultdict(list)
    per_level_agent3 = defaultdict(list)

    exp = csv_dir.name
    for csv_path in filtered_participant_paths(exp):
        file_data = parse_participant_csv(csv_path)
        for level, counts in file_data.items():
            if level in SKIP_LEVELS or any(level.startswith(pfx) for pfx in SKIP_PREFIXES):
                continue
            mapped = _map_level_name(level)
            activations = float(counts.get("activation_count", 0.0))
            if activations <= 0:
                continue
            per_level_agent2[mapped].append(float(counts.get("agent2_count", 0.0)) / activations)
            per_level_agent3[mapped].append(float(counts.get("agent3_count", 0.0)) / activations)

    out = {}
    for level in set(per_level_agent2) | set(per_level_agent3):
        a2 = per_level_agent2.get(level, [])
        a3 = per_level_agent3.get(level, [])
        out[level] = {
            "agent2_mean": _safe_mean(a2),
            "agent3_mean": _safe_mean(a3),
            "agent2_sem": _safe_sem(a2),
            "agent3_sem": _safe_sem(a3),
            "agent2_samples": list(a2),
            "agent3_samples": list(a3),
        }
    return out


def _build_exp12_data(social_root: Path, exp: str) -> dict:
    dp_root = social_root / "data_processing"
    model_root = social_root / "model_outputs"

    if exp == "exp1":
        human = _build_human_stats_exp12(dp_root / "data_processed/exp1")
        if not human:
            human = _load_results_py(dp_root / "results/current/results_dict_exp1_50.py")
        full = _load_json(model_root / "experiments/exp1/steps_dict.json")
        social = _load_json(model_root / "baselines/exp1/step_dict_social_mentalizing.json")
        nonmental = _load_json(
            model_root / "baselines/exp1/step_dict_rational_non_mentalizing.json"
        )
        naive = _load_json(model_root / "baselines/exp1/step_dict_naive_observer.json")
        full_map = {f"mod_{k}": v for k, v in full.items()}
        social_map = {f"mod_{k}_1": v for k, v in social.items()}
        non_map = {f"mod_{k}_1": v for k, v in nonmental.items()}
        naive_map = {f"mod_{k}_1": v for k, v in naive.items()}
    elif exp == "exp2":
        human = _build_human_stats_exp12(dp_root / "data_processed/exp2")
        if not human:
            human = _load_results_py(dp_root / "results/current/results_dict_merged_exp2.py")
        full_map = _load_json(model_root / "experiments/exp2/steps_dict.json")
        social_map = _load_json(model_root / "baselines/exp2/step_dict_social_mentalizing.json")
        non_map = _load_json(
            model_root / "baselines/exp2/step_dict_rational_non_mentalizing.json"
        )
        naive_map = _load_json(model_root / "baselines/exp2/step_dict_naive_observer.json")
    else:
        raise ValueError(f"unsupported exp: {exp}")

    levels = set(human) | set(full_map) | set(social_map) | set(non_map) | set(naive_map)
    out = {}
    for level in levels:
        h = human.get(level, {})
        human_sem = h.get("raw_sem")
        if human_sem is None:
            samples = h.get("samples") or []
            if samples:
                human_sem = _safe_sem([float(v) for v in samples])
            else:
                human_sem = 0.0
        out[level] = {
            "human": h.get("mean_observe_per_activation"),
            "human_sem": human_sem,
            "human_samples": h.get("samples"),
            "full": full_map.get(level),
            "social": social_map.get(level),
            "nonmental": non_map.get(level),
            "naive": naive_map.get(level),
        }
    return out


def _build_exp34_data(social_root: Path, exp: str) -> dict:
    dp_root = social_root / "data_processing"
    model_root = social_root / "model_outputs"

    if exp == "exp3":
        full = _load_json(model_root / "experiments/exp3/steps_dict.json")
        social = _load_json(
            model_root / "baselines/exp3/step_dict_social_mentalizing_until_one_converges.json"
        )
        nonmental = _load_json(
            model_root / "baselines/exp3/step_dict_rational_non_mentalizing.json"
        )
        naive = _load_json(model_root / "baselines/exp3/step_dict_naive_observer.json")
        human = _build_human_stats_exp34(dp_root / "data_processed/exp3")
    elif exp == "exp4":
        full = _load_json(model_root / "experiments/exp4/steps_dict.json")
        social = _load_json(
            model_root / "baselines/exp4/step_dict_social_mentalizing_until_one_converges.json"
        )
        nonmental = _load_json(
            model_root / "baselines/exp4/step_dict_rational_non_mentalizing.json"
        )
        naive = _load_json(model_root / "baselines/exp4/step_dict_naive_observer.json")
        human = _build_human_stats_exp34(dp_root / "data_processed/exp4")
    else:
        raise ValueError(f"unsupported exp: {exp}")

    def model_pair(d: dict, key: str) -> tuple[float | None, float | None]:
        item = d.get(key)
        if not isinstance(item, dict):
            return None, None
        return item.get("agent2_count"), item.get("agent3_count")

    levels = set(human) | set(full) | set(social) | set(nonmental) | set(naive)
    out = {}
    for level in levels:
        h = human.get(level, {})
        f2, f3 = model_pair(full, level)
        s2, s3 = model_pair(social, level)
        n2, n3 = model_pair(nonmental, level)
        z2, z3 = model_pair(naive, level)
        out[level] = {
            "human_agent2": h.get("agent2_mean"),
            "human_agent3": h.get("agent3_mean"),
            "human_sem_agent2": h.get("agent2_sem", 0.0),
            "human_sem_agent3": h.get("agent3_sem", 0.0),
            "human_samples_agent2": h.get("agent2_samples"),
            "human_samples_agent3": h.get("agent3_samples"),
            "full_agent2": f2,
            "full_agent3": f3,
            "social_agent2": s2,
            "social_agent3": s3,
            "nonmental_agent2": n2,
            "nonmental_agent3": n3,
            "naive_agent2": z2,
            "naive_agent3": z3,
        }
    return out


def _style_axis(ax, ylabel: str | None = "Number of\nObservations") -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(BOX)
        spine.set_linewidth(0.9)
    ax.grid(False)
    ax.set_ylabel(ylabel or "", fontsize=24)
    ax.tick_params(axis="x", labelsize=20, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=20, length=0)


def _draw_na_bar(ax, x_pos: float, width: float) -> None:
    ax.bar(
        x_pos,
        0.05,
        width=width,
        color="#efefef",
        edgecolor="#7a1d1d",
        linewidth=1.2,
        hatch="//",
    )
    ax.text(
        x_pos,
        0.18,
        "N/A",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#7a1d1d",
        fontweight="bold",
    )


def _draw_na_point(ax, x_pos: float) -> None:
    ax.scatter([x_pos], [0.05], marker="x", s=54, color="#7a1d1d", linewidths=1.8, zorder=3)
    ax.text(
        x_pos,
        0.14,
        "N/A",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#7a1d1d",
        fontweight="bold",
    )


def _draw_value(ax, x_pos: float, value: float, width: float, color: str, plot_style: str) -> None:
    if plot_style in ("point-interval", "human-dots"):
        ax.vlines(x_pos, 0, value, color=color, alpha=0.25, linewidth=1.6, zorder=1)
        ax.scatter([x_pos], [value], s=42, color=color, edgecolors="white", linewidths=0.6, zorder=3)
    else:
        ax.bar(x_pos, value, width=width, color=color, edgecolor="none")


def _draw_na(ax, x_pos: float, width: float, plot_style: str) -> None:
    if plot_style in ("point-interval", "human-dots"):
        _draw_na_point(ax, x_pos)
    else:
        _draw_na_bar(ax, x_pos, width)


def _draw_human_dots(ax, x_pos: float, samples: list[float] | None, color: str, seed_key: str) -> tuple[float, float]:
    vals = [float(v) for v in (samples or [])]
    if not vals:
        return 0.0, 0.0

    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    jitter = 0.085
    xs = [x_pos + rng.uniform(-jitter, jitter) for _ in vals]
    ax.scatter(xs, vals, s=18, color=color, alpha=0.42, edgecolors="none", zorder=2)

    if len(vals) >= 2:
        q1, med, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    else:
        q1 = med = q3 = vals[0]
    ax.vlines(x_pos, q1, q3, color="black", linewidth=1.8, zorder=4)
    ax.hlines(med, x_pos - 0.075, x_pos + 0.075, color="black", linewidth=1.8, zorder=4)
    ax.scatter([x_pos], [statistics.mean(vals)], s=52, color=color, edgecolors="white", linewidths=0.7, zorder=5)
    return max(vals), q3


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned[:120] if cleaned else "plot"


def _normalize_custom_level_id(exp: str, raw_level_id: str) -> str:
    level_id = str(raw_level_id or "").strip()
    if not level_id:
        raise ValueError("missing level id")

    if exp == "exp1":
        if re.match(r"^mod_s\d+_\d+$", level_id):
            return level_id
        if re.match(r"^mod_s\d+$", level_id):
            return f"{level_id}_1"
        if re.match(r"^s\d+_\d+$", level_id):
            return f"mod_{level_id}"
        if re.match(r"^s\d+$", level_id):
            return f"mod_{level_id}_1"
        raise ValueError(f"unrecognized exp1 id: {level_id}")

    if exp == "exp2":
        if re.match(r"^s\d+_\d+$", level_id):
            return level_id
        if re.match(r"^s\d+$", level_id):
            return f"{level_id}_1"
        raise ValueError(f"unrecognized exp2 id: {level_id}")

    if exp in ("exp3", "exp4"):
        if re.match(r"^sm[\w\d]+_scenario\d+$", level_id):
            return level_id
        m = re.match(r"^(sm[\w\d]+)_(\d+)$", level_id)
        if m:
            return f"{m.group(1)}_scenario{m.group(2)}"
        if re.match(r"^sm[\w\d]+$", level_id):
            return f"{level_id}_scenario1"
        raise ValueError(f"unrecognized {exp} id: {level_id}")

    raise ValueError(f"unsupported experiment: {exp}")


def _parse_custom_stimulus(exp: str, raw_stimulus, label: str) -> tuple[str, int | None]:
    if isinstance(raw_stimulus, str):
        stimulus = {"id": raw_stimulus}
    elif isinstance(raw_stimulus, dict):
        stimulus = raw_stimulus
    else:
        raise ValueError(f"{label}: entry must be an object or string")

    raw_id = stimulus.get("id") or stimulus.get("jsonLevelId") or stimulus.get("levelId")
    if not raw_id:
        raise ValueError(f"{label}: missing id/jsonLevelId/levelId")

    level_id = _normalize_custom_level_id(exp, str(raw_id))
    raw_timestamp = stimulus.get("timestamp")
    timestamp = None
    if raw_timestamp not in (None, ""):
        timestamp = int(raw_timestamp)
        if timestamp < 0:
            raise ValueError(f"{label}: timestamp must be >= 0")
    return level_id, timestamp


def _render_exp12_pair_plot(
    row_left: dict,
    row_right: dict,
    out_path: Path,
    plot_style: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
) -> None:
    values_left = [
        row_left.get("human"),
        row_left.get("full"),
        row_left.get("social"),
        row_left.get("nonmental"),
        row_left.get("naive"),
    ]
    values_right = [
        row_right.get("human"),
        row_right.get("full"),
        row_right.get("social"),
        row_right.get("nonmental"),
        row_right.get("naive"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.34
    ymax = 1.0

    for i in xs:
        x1 = i - width / 2
        x2 = i + width / 2

        if i == 0 and plot_style == "human-dots":
            continue

        v1 = values_left[i]
        v2 = values_right[i]
        if v1 is None:
            _draw_na(ax, x1, width, plot_style)
        else:
            _draw_value(ax, x1, v1, width, left_color, plot_style)
            ymax = max(ymax, float(v1))

        if v2 is None:
            _draw_na(ax, x2, width, plot_style)
        else:
            _draw_value(ax, x2, v2, width, right_color, plot_style)
            ymax = max(ymax, float(v2))

    h1 = row_left.get("human")
    h2 = row_right.get("human")
    sem1 = row_left.get("human_sem", 0.0) or 0.0
    sem2 = row_right.get("human_sem", 0.0) or 0.0
    samples1 = row_left.get("human_samples")
    samples2 = row_right.get("human_samples")

    if plot_style == "human-dots":
        max1, q31 = _draw_human_dots(ax, 0 - width / 2, samples1, left_color, f"{left_label}:left")
        max2, q32 = _draw_human_dots(ax, 0 + width / 2, samples2, right_color, f"{right_label}:right")
        if not samples1 and h1 is not None:
            _draw_value(ax, 0 - width / 2, float(h1), width, left_color, plot_style)
        if not samples2 and h2 is not None:
            _draw_value(ax, 0 + width / 2, float(h2), width, right_color, plot_style)
        ymax = max(ymax, max1, q31, max2, q32)
    else:
        if h1 is not None:
            ax.errorbar([0 - width / 2], [h1], yerr=[sem1], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
            ymax = max(ymax, float(h1 + sem1))
        if h2 is not None:
            ax.errorbar([0 + width / 2], [h2], yerr=[sem2], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
            ymax = max(ymax, float(h2 + sem2))

    _style_axis(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_LABELS)
    for label in ax.get_xticklabels():
        label.set_fontsize(22)
        label.set_linespacing(0.92)
    ax.legend(
        [left_label, right_label],
        loc="upper left",
        fontsize=22,
        frameon=True,
        facecolor=BG,
        edgecolor=BOX,
    )
    ax.set_ylim(0, ymax * 1.16 + 0.20)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.34, top=0.92)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_exp12_single_plot(
    row: dict,
    out_path: Path,
    plot_style: str,
    show_ylabel: bool,
) -> None:
    values = [row.get("human"), row.get("full"), row.get("social"), row.get("nonmental"), row.get("naive")]
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.56
    ymax = 1.0

    human = row.get("human")
    human_sem = row.get("human_sem", 0.0) or 0.0
    human_samples = row.get("human_samples")

    for i, val in enumerate(values):
        if i == 0 and plot_style == "human-dots":
            continue
        if val is None:
            _draw_na(ax, i, width, plot_style)
            continue
        if plot_style == "bar":
            ax.bar(
                i,
                val,
                width=width,
                color=EXP12_SPLIT_BLUE_SHADES[i],
                edgecolor="#23406b",
                linewidth=0.6,
                hatch=EXP12_SPLIT_HATCHES[i],
            )
        else:
            _draw_value(ax, i, val, width, PAIR_BLUE_DARK, plot_style)
        ymax = max(ymax, float(val))

    if plot_style == "human-dots":
        if human is None and human_samples:
            human = _safe_mean([float(v) for v in human_samples])
        max_sample, q3 = _draw_human_dots(ax, 0, human_samples, PAIR_BLUE_DARK, "exp12:single")
        if not human_samples and human is not None:
            _draw_value(ax, 0, float(human), width, PAIR_BLUE_DARK, plot_style)
        ymax = max(ymax, max_sample, q3)
        if human is not None:
            ymax = max(ymax, float(human))
    elif human is not None:
        ax.errorbar([0], [human], yerr=[human_sem], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(human + human_sem))

    _style_axis(ax, "Number of\nObservations" if show_ylabel else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(EXP12_SPLIT_LABELS, rotation=28, ha="right", rotation_mode="anchor")
    for label in ax.get_xticklabels():
        label.set_fontsize(20)
        label.set_linespacing(0.92)
    ax.set_ylim(0, ymax * 1.16 + 0.20)
    fig.subplots_adjust(left=0.15 if show_ylabel else 0.09, right=0.99, bottom=0.42, top=0.92)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_exp34_agent_plot(
    level: str,
    row: dict,
    out_path: Path,
    agent: int,
    plot_style: str,
) -> None:
    if agent == 2:
        values = [
            row.get("human_agent2"),
            row.get("full_agent2"),
            row.get("social_agent2"),
            row.get("nonmental_agent2"),
            row.get("naive_agent2"),
        ]
        human = row.get("human_agent2")
        human_sem = row.get("human_sem_agent2", 0.0) or 0.0
        human_samples = row.get("human_samples_agent2")
        color = AGENT2_BLUE_S1
        agent_label = "Agent 2"
        bar_colors = AGENT2_BLUE_SHADES
    elif agent == 3:
        values = [
            row.get("human_agent3"),
            row.get("full_agent3"),
            row.get("social_agent3"),
            row.get("nonmental_agent3"),
            row.get("naive_agent3"),
        ]
        human = row.get("human_agent3")
        human_sem = row.get("human_sem_agent3", 0.0) or 0.0
        human_samples = row.get("human_samples_agent3")
        color = AGENT3_GREEN_S1
        agent_label = "Agent 3"
        bar_colors = AGENT3_GREEN_SHADES
    else:
        raise ValueError(f"unsupported agent id: {agent}")

    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.56
    ymax = 1.0

    for i, val in enumerate(values):
        if i == 0 and plot_style == "human-dots":
            continue
        if val is None:
            _draw_na(ax, i, width, plot_style)
        else:
            if plot_style == "bar":
                ax.bar(
                    i,
                    val,
                    width=width,
                    color=bar_colors[i],
                    edgecolor="#23406b" if agent == 2 else "#1f5a38",
                    linewidth=0.6,
                    hatch=MODEL_HATCHES[i],
                )
            else:
                _draw_value(ax, i, val, width, color, plot_style)
            ymax = max(ymax, float(val))

    if plot_style == "human-dots":
        if human is None and human_samples:
            human = _safe_mean([float(v) for v in human_samples])
        max_sample, q3 = _draw_human_dots(ax, 0, human_samples, color, f"{level}:agent{agent}")
        if not human_samples and human is not None:
            _draw_value(ax, 0, float(human), width, color, plot_style)
        ymax = max(ymax, max_sample, q3)
        if human is not None:
            ymax = max(ymax, float(human))
    elif human is not None:
        ax.errorbar([0], [human], yerr=[human_sem], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(human + human_sem))

    _style_axis(ax)
    ax.set_xticks(xs)
    if agent == 3:
        ax.set_xticklabels(ABBREV_GROUP_LABELS)
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0, ymax * 1.16 + 0.20)
    if plot_style != "bar":
        ax.text(
            0.98,
            0.98,
            agent_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=22,
            fontweight="bold",
            color=color,
        )
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.28, top=0.90)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _build_selected_levels(stimuli_config: dict, exp: str) -> list[tuple[str, int | None]]:
    raw_items = stimuli_config.get(exp, [])
    if not isinstance(raw_items, list):
        raise ValueError(f"stimuli config key '{exp}' must be a list")

    selected = []
    for idx, item in enumerate(raw_items, start=1):
        selected.append(_parse_custom_stimulus(exp, item, f"{exp}[{idx - 1}]"))
    return selected


def _alpha_label(n: int) -> str:
    if n <= 0:
        return "A"
    out = []
    x = n
    while x > 0:
        x, r = divmod(x - 1, 26)
        out.append(chr(ord("A") + r))
    return "".join(reversed(out))


def _build_ordered_labels(stimuli_config: dict) -> dict[str, list[str]]:
    label_map: dict[str, list[str]] = {}
    counter = 1
    for exp in ("exp1", "exp2", "exp3", "exp4"):
        raw_items = stimuli_config.get(exp, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"stimuli config key '{exp}' must be a list")
        labels = []
        for _ in raw_items:
            labels.append(_alpha_label(counter))
            counter += 1
        label_map[exp] = labels
    return label_map


def _timestamp_suffix(timestamp: int | None) -> str:
    return f"_t{timestamp}" if timestamp is not None else ""


def _generate_exp12_plots(
    exp: str,
    selected: list[tuple[str, int | None]],
    labels: list[str],
    level_data: dict,
    output_dir: Path,
    plot_style: str,
) -> list[Path]:
    if len(selected) != 2:
        raise ValueError(f"{exp} requires exactly 2 selected items, found {len(selected)}")
    if len(labels) != 2:
        raise ValueError(f"{exp} requires exactly 2 labels, found {len(labels)}")

    outputs: list[Path] = []
    for idx, ((level, timestamp), _panel_letter) in enumerate(zip(selected, labels), start=1):
        if level not in level_data:
            raise KeyError(f"missing level data for {level}")
        out_path = output_dir / f"{exp}_{idx:02d}_{_slug(level)}{_timestamp_suffix(timestamp)}.png"
        _render_exp12_single_plot(
            level_data[level],
            out_path,
            plot_style,
            show_ylabel=(exp == "exp1" and idx == 1),
        )
        outputs.append(out_path)
    return outputs


def _generate_exp34_plots(
    exp: str,
    selected: list[tuple[str, int | None]],
    labels: list[str],
    level_data: dict,
    output_dir: Path,
    plot_style: str,
) -> list[Path]:
    outputs: list[Path] = []
    if len(labels) != len(selected):
        raise ValueError(f"{exp} label count does not match selected item count")
    for idx, ((level, timestamp), _panel_letter) in enumerate(zip(selected, labels), start=1):
        if level not in level_data:
            raise KeyError(f"missing level data for {level}")
        for agent in (2, 3):
            out_path = output_dir / (
                f"{exp}_{idx:02d}_{_slug(level)}{_timestamp_suffix(timestamp)}_agent{agent}.png"
            )
            _render_exp34_agent_plot(
                level,
                level_data[level],
                out_path,
                agent=agent,
                plot_style=plot_style,
            )
            outputs.append(out_path)
    return outputs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate custom-stimuli observation plot PNGs in config order."
    )
    parser.add_argument(
        "--stimuli-config",
        type=Path,
        default=script_dir / "custom_stimuli_timestamps.json",
    )
    parser.add_argument(
        "--social-root",
        type=Path,
        default=script_dir.parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "custom_stimuli_batch_1" / "plots",
    )
    parser.add_argument(
        "--exp",
        choices=["all", "exp1", "exp2", "exp3", "exp4"],
        default="all",
    )
    parser.add_argument(
        "--plot-style",
        choices=["bar", "point-interval", "human-dots"],
        default="bar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stimuli_config = _load_json(args.stimuli_config.resolve())
    social_root = args.social_root.resolve()
    output_dir = args.output_dir.resolve()

    if not social_root.exists():
        raise FileNotFoundError(f"social root not found: {social_root}")
    if not isinstance(stimuli_config, dict):
        raise ValueError("stimuli config must be a JSON object with exp keys")

    exps = [args.exp] if args.exp != "all" else ["exp1", "exp2", "exp3", "exp4"]
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_level_data = {
        "exp1": _build_exp12_data(social_root, "exp1"),
        "exp2": _build_exp12_data(social_root, "exp2"),
        "exp3": _build_exp34_data(social_root, "exp3"),
        "exp4": _build_exp34_data(social_root, "exp4"),
    }
    exp_labels = _build_ordered_labels(stimuli_config)

    all_outputs: list[Path] = []
    for exp in exps:
        selected = _build_selected_levels(stimuli_config, exp)
        if exp in ("exp1", "exp2"):
            outputs = _generate_exp12_plots(
                exp,
                selected,
                exp_labels[exp],
                exp_level_data[exp],
                output_dir,
                args.plot_style,
            )
        else:
            outputs = _generate_exp34_plots(
                exp,
                selected,
                exp_labels[exp],
                exp_level_data[exp],
                output_dir,
                args.plot_style,
            )
        all_outputs.extend(outputs)
        print(f"{exp}: wrote {len(outputs)} plot(s)")
        for out_path in outputs:
            print(f"  {out_path}")

    print(f"done: wrote {len(all_outputs)} total plot(s) to {output_dir}")


if __name__ == "__main__":
    main()
