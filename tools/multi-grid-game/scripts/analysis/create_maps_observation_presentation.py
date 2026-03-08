#!/usr/bin/env python3
# /// script
# dependencies = [
#   "python-pptx",
#   "matplotlib",
# ]
# ///

"""
Create stylized PowerPoint decks from map images with observation bar plots.

Outputs 4 decks by default:
  - maps_observations_exp1.pptx
  - maps_observations_exp2.pptx
  - maps_observations_exp3.pptx
  - maps_observations_exp4.pptx

Rules:
  - Exp1: one map per slide
  - Exp2/Exp3/Exp4: scenario1 + scenario2 paired on one slide (side-by-side)
  - Plot box width matches the rendered map width for each panel
  - Human bars include error bars
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# ---- style ----
BG = "#ffffff"
BOX = "#707070"
SERIES_A1 = "#de6b8a"  # pink
SERIES_A2 = "#f0b35a"  # orange
AGENT2_BLUE_S1 = "#2f6df3"
AGENT2_BLUE_S2 = "#8fb7ff"
AGENT3_GREEN_S1 = "#22a15a"
AGENT3_GREEN_S2 = "#8dd9ad"

GROUP_KEYS = ["human", "full", "social", "nonmental", "naive"]
GROUP_LABELS = [
    "Human",
    "Rational\nMentalizing",
    "Social\nMentalizing",
    "Rational\nNon-mentalizing",
    "Naive",
]

LEVEL_RE = re.compile(r"^Level:\s*(.+?)\s*$", re.IGNORECASE)
SKIP_LEVELS = {"comprehension_check", "experiment", "s111_1"}
SKIP_PREFIXES = ("sm111_", "sm112_")


@dataclass
class PanelSpec:
    level: str
    image_path: Path


@dataclass
class SlideSpec:
    exp: str
    title: str
    left: PanelSpec
    right: PanelSpec | None = None


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_results_py(path: Path) -> dict:
    ns = {}
    exec(path.read_text(encoding="utf-8"), ns)
    return ns.get("results_dict", {})


def _map_level_name(human_level: str) -> str:
    m = re.match(r"^(sm\d+)_(\d+)$", human_level)
    if m:
        return f"{m.group(1)}_scenario{m.group(2)}"

    m = re.match(r"^(sm\d+)_true$", human_level)
    if m:
        return f"{m.group(1)}_scenario1"

    return human_level


def _parse_multi_agent_csv(path: Path) -> dict:
    level_data = defaultdict(
        lambda: {
            "agent2_count": 0,
            "agent3_count": 0,
            "activation_count": 0,
        }
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
                    if "OBSERVE" in type_field:
                        agent_id = (row[5] or "").strip()
                        if agent_id in ("2", "2.0"):
                            level_data[current_level]["agent2_count"] += 1
                        elif agent_id in ("3", "3.0"):
                            level_data[current_level]["agent3_count"] += 1

    return dict(level_data)


def _build_human_stats_exp34(csv_dir: Path) -> dict:
    per_level_agent2 = defaultdict(list)
    per_level_agent3 = defaultdict(list)

    for csv_path in sorted(csv_dir.glob("*.csv")):
        file_data = _parse_multi_agent_csv(csv_path)
        for level, counts in file_data.items():
            if level in SKIP_LEVELS or any(level.startswith(pfx) for pfx in SKIP_PREFIXES):
                continue

            mapped = _map_level_name(level)
            activations = counts["activation_count"]
            if activations <= 0:
                continue

            per_level_agent2[mapped].append(counts["agent2_count"] / activations)
            per_level_agent3[mapped].append(counts["agent3_count"] / activations)

    out = {}
    for level in set(per_level_agent2) | set(per_level_agent3):
        a2 = per_level_agent2.get(level, [])
        a3 = per_level_agent3.get(level, [])
        out[level] = {
            "agent2_mean": _safe_mean(a2),
            "agent3_mean": _safe_mean(a3),
            "agent2_sd": _safe_sd(a2),
            "agent3_sd": _safe_sd(a3),
        }
    return out


def _build_exp12_data(social_root: Path, exp: str) -> dict:
    dp_root = social_root / "data_processing"

    if exp == "exp1":
        human = _load_results_py(dp_root / "results/current/results_dict_exp1_50.py")
        full = _load_json(social_root / "steps.dict.json")
        social = _load_json(social_root / "scripts/baselines/exp1/step_dict_mentalize_exp1.json")
        nonmental = _load_json(social_root / "scripts/baselines/exp1/step_dict_nonmentalize_exp1.json")
        naive = _load_json(social_root / "scripts/baselines/exp1/step_dict_naive_exp1.json")

        full_map = {k.replace("_ascii", "_1"): v for k, v in full.items()}
        social_map = {f"mod_{k}_1": v for k, v in social.items()}
        non_map = {f"mod_{k}_1": v for k, v in nonmental.items()}
        naive_map = {f"mod_{k}_1": v for k, v in naive.items()}

    elif exp == "exp2":
        human = _load_results_py(dp_root / "results/current/results_dict_merged_exp2.py")
        full_map = _load_json(social_root / "steps_exp2.json")
        social_map = _load_json(social_root / "scripts/baselines/exp2/step_dict_mentalize_exp2.json")
        non_map = _load_json(social_root / "scripts/baselines/exp2/step_dict_nonmentalize_exp2.json")
        naive_map = _load_json(social_root / "scripts/baselines/exp2/step_dict_naive_exp2.json")

    else:
        raise ValueError(exp)

    levels = set(human) | set(full_map) | set(social_map) | set(non_map) | set(naive_map)

    out = {}
    for level in levels:
        h = human.get(level, {})
        out[level] = {
            "human": h.get("mean_observe_per_activation"),
            "human_sd": h.get("bootstrap_sd", 0.0),
            "full": full_map.get(level),
            "social": social_map.get(level),
            "nonmental": non_map.get(level),
            "naive": naive_map.get(level),
        }
    return out


def _build_exp34_data(social_root: Path, exp: str) -> dict:
    dp_root = social_root / "data_processing"

    if exp == "exp3":
        full = _load_json(social_root / "steps_dict_exp3.json")
        social = _load_json(social_root / "scripts/baselines/exp3/step_dict_mentalize_exp3.json")
        nonmental = _load_json(social_root / "scripts/baselines/exp3/step_dict_nonmentalize_exp3_v2.json")
        naive = _load_json(social_root / "scripts/baselines/exp3/step_dict_naive_exp3.json")
        human = _build_human_stats_exp34(dp_root / "data_processed/exp3")
    elif exp == "exp4":
        full = _load_json(social_root / "scripts/experiments/experiment_outputs/steps_dict_exp4_020126_2.json")
        social = _load_json(social_root / "step_dict_mentalize_exp4.json")
        nonmental = _load_json(social_root / "scripts/baselines/exp4/step_dict_nonmentalize_exp4_v2.json")
        naive = _load_json(social_root / "scripts/baselines/exp4/step_dict_naive_exp4.json")
        human = _build_human_stats_exp34(dp_root / "data_processed/exp4")
    else:
        raise ValueError(exp)

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
            "human_sd_agent2": h.get("agent2_sd", 0.0),
            "human_sd_agent3": h.get("agent3_sd", 0.0),
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


def _style_axis(ax, title: str) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(BOX)
        spine.set_linewidth(0.9)
    ax.grid(False)
    ax.set_ylabel("Number of\nObservations", fontsize=16)
    ax.tick_params(axis="x", labelsize=13, length=0)
    ax.tick_params(axis="y", labelsize=12, length=0)


def _add_map_title_above_emu(slide, text: str, map_left_emu: int, map_top_emu: int, map_width_emu: int) -> None:
    title_h = Inches(0.34)
    title_top = int(max(Inches(0.34), map_top_emu - Inches(0.28)))
    box = slide.shapes.add_textbox(map_left_emu, title_top, map_width_emu, title_h)
    tf = box.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.text = text
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(18)


def _draw_na_bar(ax, x_pos: float, width: float) -> None:
    ax.bar(x_pos, 0.05, width=width, color="#efefef", edgecolor="#7a1d1d", linewidth=1.2, hatch="//")
    ax.text(x_pos, 0.2, "N/A", ha="center", va="bottom", fontsize=9, color="#7a1d1d", fontweight="bold")


def _render_exp12_plot(level: str, row: dict, out_path: Path) -> None:
    values = [row.get("human"), row.get("full"), row.get("social"), row.get("nonmental"), row.get("naive")]
    fig, ax = plt.subplots(figsize=(11.2, 2.95), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.62
    ymax = 1.0

    for i, val in enumerate(values):
        if val is None:
            _draw_na_bar(ax, i, width)
            continue
        ax.bar(i, val, width=width, color=SERIES_A1, edgecolor="none")
        ymax = max(ymax, float(val))

    human = row.get("human")
    human_sd = row.get("human_sd", 0.0) or 0.0
    if human is not None:
        ax.errorbar([0], [human], yerr=[human_sd], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(human + human_sd))

    _style_axis(ax, level)
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylim(0, ymax * 1.28 + 0.3)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.31, top=0.86)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_exp34_plot(level: str, row: dict, out_path: Path) -> None:
    a1 = [
        row.get("human_agent2"),
        row.get("full_agent2"),
        row.get("social_agent2"),
        row.get("nonmental_agent2"),
        row.get("naive_agent2"),
    ]
    a2 = [
        row.get("human_agent3"),
        row.get("full_agent3"),
        row.get("social_agent3"),
        row.get("nonmental_agent3"),
        row.get("naive_agent3"),
    ]

    fig, ax = plt.subplots(figsize=(11.2, 2.95), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.34
    ymax = 1.0

    for i in xs:
        x1 = i - width / 2
        x2 = i + width / 2

        if a1[i] is None:
            _draw_na_bar(ax, x1, width)
        else:
            ax.bar(x1, a1[i], width=width, color=AGENT2_BLUE_S1, edgecolor="none")
            ymax = max(ymax, float(a1[i]))

        if a2[i] is None:
            _draw_na_bar(ax, x2, width)
        else:
            ax.bar(x2, a2[i], width=width, color=AGENT3_GREEN_S1, edgecolor="none")
            ymax = max(ymax, float(a2[i]))

    h2 = row.get("human_agent2")
    h3 = row.get("human_agent3")
    sd2 = row.get("human_sd_agent2", 0.0) or 0.0
    sd3 = row.get("human_sd_agent3", 0.0) or 0.0

    if h2 is not None:
        ax.errorbar([0 - width / 2], [h2], yerr=[sd2], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(h2 + sd2))
    if h3 is not None:
        ax.errorbar([0 + width / 2], [h3], yerr=[sd3], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(h3 + sd3))

    _style_axis(ax, level)
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_LABELS)
    ax.legend(
        ["Agent 2 (Blue)", "Agent 3 (Green)"],
        loc="upper left",
        fontsize=10,
        frameon=True,
        facecolor=BG,
        edgecolor=BOX,
    )
    ax.set_ylim(0, ymax * 1.28 + 0.4)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.31, top=0.86)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_exp12_combined_plot(
    base_level: str,
    row_s1: dict,
    row_s2: dict,
    out_path: Path,
) -> None:
    values_s1 = [row_s1.get("human"), row_s1.get("full"), row_s1.get("social"), row_s1.get("nonmental"), row_s1.get("naive")]
    values_s2 = [row_s2.get("human"), row_s2.get("full"), row_s2.get("social"), row_s2.get("nonmental"), row_s2.get("naive")]

    fig, ax = plt.subplots(figsize=(13.0, 3.9), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.34
    ymax = 1.0

    for i in xs:
        x1 = i - width / 2
        x2 = i + width / 2
        v1 = values_s1[i]
        v2 = values_s2[i]

        if v1 is None:
            _draw_na_bar(ax, x1, width)
        else:
            ax.bar(x1, v1, width=width, color=SERIES_A1, edgecolor="none")
            ymax = max(ymax, float(v1))

        if v2 is None:
            _draw_na_bar(ax, x2, width)
        else:
            ax.bar(x2, v2, width=width, color=SERIES_A2, edgecolor="none")
            ymax = max(ymax, float(v2))

    h1 = row_s1.get("human")
    h2 = row_s2.get("human")
    sd1 = row_s1.get("human_sd", 0.0) or 0.0
    sd2 = row_s2.get("human_sd", 0.0) or 0.0
    if h1 is not None:
        ax.errorbar([0 - width / 2], [h1], yerr=[sd1], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(h1 + sd1))
    if h2 is not None:
        ax.errorbar([0 + width / 2], [h2], yerr=[sd2], fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ymax = max(ymax, float(h2 + sd2))

    _style_axis(ax, f"{base_level} | Scenario 1 vs Scenario 2")
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_LABELS)
    ax.legend(["Scenario 1", "Scenario 2"], loc="upper left", fontsize=10, frameon=True, facecolor=BG, edgecolor=BOX)
    ax.set_ylim(0, ymax * 1.25 + 0.35)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.28, top=0.86)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_exp34_combined_plot(
    base_level: str,
    row_s1: dict,
    row_s2: dict,
    out_path: Path,
) -> None:
    s1_a1 = [
        row_s1.get("human_agent2"),
        row_s1.get("full_agent2"),
        row_s1.get("social_agent2"),
        row_s1.get("nonmental_agent2"),
        row_s1.get("naive_agent2"),
    ]
    s1_a2 = [
        row_s1.get("human_agent3"),
        row_s1.get("full_agent3"),
        row_s1.get("social_agent3"),
        row_s1.get("nonmental_agent3"),
        row_s1.get("naive_agent3"),
    ]
    s2_a1 = [
        row_s2.get("human_agent2"),
        row_s2.get("full_agent2"),
        row_s2.get("social_agent2"),
        row_s2.get("nonmental_agent2"),
        row_s2.get("naive_agent2"),
    ]
    s2_a2 = [
        row_s2.get("human_agent3"),
        row_s2.get("full_agent3"),
        row_s2.get("social_agent3"),
        row_s2.get("nonmental_agent3"),
        row_s2.get("naive_agent3"),
    ]

    fig, ax = plt.subplots(figsize=(13.4, 4.1), dpi=220)
    fig.patch.set_facecolor(BG)

    xs = list(range(len(GROUP_KEYS)))
    width = 0.18
    ymax = 1.0

    for i in xs:
        slots = [
            (i - 1.5 * width, s1_a1[i], AGENT2_BLUE_S1),
            (i - 0.5 * width, s1_a2[i], AGENT3_GREEN_S1),
            (i + 0.5 * width, s2_a1[i], AGENT2_BLUE_S2),
            (i + 1.5 * width, s2_a2[i], AGENT3_GREEN_S2),
        ]
        for x_pos, value, color in slots:
            if value is None:
                _draw_na_bar(ax, x_pos, width)
            else:
                ax.bar(x_pos, value, width=width, color=color, edgecolor="none")
                ymax = max(ymax, float(value))

    human_points = [
        (0 - 1.5 * width, row_s1.get("human_agent2"), row_s1.get("human_sd_agent2", 0.0) or 0.0),
        (0 - 0.5 * width, row_s1.get("human_agent3"), row_s1.get("human_sd_agent3", 0.0) or 0.0),
        (0 + 0.5 * width, row_s2.get("human_agent2"), row_s2.get("human_sd_agent2", 0.0) or 0.0),
        (0 + 1.5 * width, row_s2.get("human_agent3"), row_s2.get("human_sd_agent3", 0.0) or 0.0),
    ]
    for x_pos, value, sd in human_points:
        if value is not None:
            ax.errorbar([x_pos], [value], yerr=[sd], fmt="none", ecolor="black", capsize=3, linewidth=1.1)
            ymax = max(ymax, float(value + sd))

    _style_axis(ax, f"{base_level} | Scenario 1 & 2 (A.1/A.2)")
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_LABELS)
    ax.legend(
        [
            "Scenario 1 - Agent 2 (Blue)",
            "Scenario 1 - Agent 3 (Green)",
            "Scenario 2 - Agent 2 (Blue)",
            "Scenario 2 - Agent 3 (Green)",
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
        facecolor=BG,
        edgecolor=BOX,
        ncol=2,
    )
    ax.set_ylim(0, ymax * 1.24 + 0.40)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.28, top=0.86)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _level_from_image(path: Path) -> str:
    stem = path.stem
    return stem.replace("stimuli_", "", 1) if stem.startswith("stimuli_") else stem


def _is_omitted_level(level: str) -> bool:
    m = re.search(r"s(?:m)?(\d{3})", level)
    return bool(m and m.group(1) in {"111", "112"})


def _paired_key(level: str, exp: str) -> tuple[str, str] | None:
    if exp == "exp2":
        m = re.match(r"^(.*)_(1|2)$", level)
        if m:
            return m.group(1), m.group(2)
    elif exp in ("exp3", "exp4"):
        m = re.match(r"^(.*)_scenario(1|2)$", level)
        if m:
            return m.group(1), m.group(2)
    return None


def _build_slide_specs(exp: str, images: list[Path]) -> list[SlideSpec]:
    if exp == "exp1":
        out = []
        for img in images:
            level = _level_from_image(img)
            out.append(SlideSpec(exp=exp, title=f"{exp.upper()} | {level}", left=PanelSpec(level=level, image_path=img)))
        return out

    grouped: dict[str, dict[str, PanelSpec]] = defaultdict(dict)
    singles: list[SlideSpec] = []

    for img in images:
        level = _level_from_image(img)
        pk = _paired_key(level, exp)
        if pk is None:
            singles.append(SlideSpec(exp=exp, title=f"{exp.upper()} | {level}", left=PanelSpec(level=level, image_path=img)))
            continue

        base, variant = pk
        grouped[base][variant] = PanelSpec(level=level, image_path=img)

    paired = []
    for base in sorted(grouped):
        variants = grouped[base]
        left = variants.get("1") or variants.get("2")
        right = variants.get("2") if variants.get("1") else None
        if left is None:
            continue
        paired.append(SlideSpec(exp=exp, title=f"{exp.upper()} | {base}", left=left, right=right))

    return paired + singles


def _add_slide_title(slide, title: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.25), Inches(0.05), Inches(12.85), Inches(0.32))
    tf = box.text_frame
    tf.text = title
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(18)


def _add_panel_label(slide, text: str, left: float, width: float) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(0.40), Inches(width), Inches(0.25))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(12)


def _add_panel(
    slide,
    prs: Presentation,
    panel: PanelSpec,
    chart_path: Path,
    left: float,
    panel_width: float,
    map_region_top: float,
    map_region_height: float,
    chart_top: float,
    chart_height: float,
    chart_width_factor: float = 2.2,
) -> None:
    # map first (fit into region)
    pic = slide.shapes.add_picture(str(panel.image_path), Inches(left), Inches(map_region_top), width=Inches(panel_width))

    map_max_h = Inches(map_region_height)
    if pic.height > map_max_h:
        scale = map_max_h / pic.height
        pic.height = int(pic.height * scale)
        pic.width = int(pic.width * scale)

    # center map in panel + map region
    panel_left_emu = Inches(left)
    panel_width_emu = Inches(panel_width)
    map_top_emu = Inches(map_region_top)
    map_h_emu = Inches(map_region_height)

    pic.left = int(panel_left_emu + (panel_width_emu - pic.width) / 2)
    pic.top = int(map_top_emu + (map_h_emu - pic.height) / 2)
    _add_map_title_above_emu(slide, panel.level, pic.left, pic.top, pic.width)

    # chart gets much wider than the map (up to panel width), without distortion.
    chart_target_w = min(panel_width_emu, int(pic.width * chart_width_factor))
    chart_left_emu = int(panel_left_emu + (panel_width_emu - chart_target_w) / 2)

    chart = slide.shapes.add_picture(
        str(chart_path),
        chart_left_emu,
        Inches(chart_top),
        width=chart_target_w,
    )

    max_chart_h = Inches(chart_height)
    if chart.height > max_chart_h:
        scale = max_chart_h / chart.height
        chart.height = int(chart.height * scale)
        chart.width = int(chart.width * scale)

    # Keep centered in panel even after any height-based scaling.
    chart.left = int(panel_left_emu + (panel_width_emu - chart.width) / 2)


def _add_map_only(
    slide,
    image_path: Path,
    left: float,
    panel_width: float,
    top: float,
    height: float,
    align: str = "center",
):
    pic = slide.shapes.add_picture(str(image_path), Inches(left), Inches(top), width=Inches(panel_width))
    max_h = Inches(height)
    if pic.height > max_h:
        scale = max_h / pic.height
        pic.height = int(pic.height * scale)
        pic.width = int(pic.width * scale)

    panel_left_emu = Inches(left)
    panel_width_emu = Inches(panel_width)
    top_emu = Inches(top)
    h_emu = Inches(height)
    if align == "left":
        pic.left = int(panel_left_emu)
    elif align == "right":
        pic.left = int(panel_left_emu + panel_width_emu - pic.width)
    else:
        pic.left = int(panel_left_emu + (panel_width_emu - pic.width) / 2)
    pic.top = int(top_emu + (h_emu - pic.height) / 2)
    return pic


def _add_chart_only(
    slide,
    chart_path: Path,
    left: float,
    width: float,
    top: float,
    height: float,
) -> None:
    chart = slide.shapes.add_picture(str(chart_path), Inches(left), Inches(top), width=Inches(width))
    max_h = Inches(height)
    if chart.height > max_h:
        scale = max_h / chart.height
        chart.height = int(chart.height * scale)
        chart.width = int(chart.width * scale)

    left_emu = Inches(left)
    width_emu = Inches(width)
    chart.left = int(left_emu + (width_emu - chart.width) / 2)


def _build_deck_for_exp(
    exp: str,
    slide_specs: list[SlideSpec],
    level_data: dict,
    output_path: Path,
    chart_dir: Path,
) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    chart_dir.mkdir(parents=True, exist_ok=True)

    for idx, spec in enumerate(slide_specs, start=1):
        slide = prs.slides.add_slide(blank)
        _add_slide_title(slide, spec.title)

        chart_left = chart_dir / f"{exp}_{spec.left.level}.png"
        row_left = level_data.get(spec.left.level, {})
        if exp in ("exp1", "exp2"):
            _render_exp12_plot(spec.left.level, row_left, chart_left)
        else:
            _render_exp34_plot(spec.left.level, row_left, chart_left)

        if spec.right is None:
            # full-width panel
            left = 0.35
            panel_w = 12.63
            _add_panel(
                slide,
                prs,
                spec.left,
                chart_left,
                left=left,
                panel_width=panel_w,
                map_region_top=0.62,
                map_region_height=4.45,
                chart_top=5.12,
                chart_height=2.20,
                chart_width_factor=2.3,
            )
        else:
            # paired layout: two maps side-by-side + one merged chart below
            margin = 0.35
            gutter = 0.64
            panel_w = (13.333 - 2 * margin - gutter) / 2
            left_x = margin
            right_x = margin + panel_w + gutter

            row_left = level_data.get(spec.left.level, {})
            row_right = level_data.get(spec.right.level, {})
            base = _paired_key(spec.left.level, exp)
            base_level = base[0] if base else spec.left.level

            chart_pair = chart_dir / f"{exp}_{base_level}_pair.png"
            if exp == "exp2":
                _render_exp12_combined_plot(base_level, row_left, row_right, chart_pair)
            else:
                _render_exp34_combined_plot(base_level, row_left, row_right, chart_pair)

            left_pic = _add_map_only(
                slide,
                spec.left.image_path,
                left=left_x,
                panel_width=panel_w,
                top=0.62,
                height=3.58,
                align="right",
            )
            _add_map_title_above_emu(slide, spec.left.level, left_pic.left, left_pic.top, left_pic.width)

            right_pic = _add_map_only(
                slide,
                spec.right.image_path,
                left=right_x,
                panel_width=panel_w,
                top=0.62,
                height=3.58,
                align="left",
            )
            _add_map_title_above_emu(slide, spec.right.level, right_pic.left, right_pic.top, right_pic.width)

            _add_chart_only(
                slide,
                chart_pair,
                left=0.35,
                width=12.63,
                top=4.20,
                height=2.95,
            )

        if idx % 20 == 0:
            print(f"  {exp}: built {idx}/{len(slide_specs)} slides")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Build stylized map+barplot PowerPoints per experiment.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=project_root / "scripts" / "generated_images",
    )
    parser.add_argument(
        "--social-root",
        type=Path,
        default=project_root.parent / "social_learning_ToM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "scripts" / "generated_images",
    )
    parser.add_argument(
        "--exp",
        choices=["all", "exp1", "exp2", "exp3", "exp4"],
        default="all",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help="Optional cap per experiment for smoke testing.",
    )
    parser.add_argument(
        "--keep-charts",
        action="store_true",
        help="Keep intermediate chart images under output-dir/barplot_cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_root = args.images_root.resolve()
    social_root = args.social_root.resolve()
    output_dir = args.output_dir.resolve()

    if not images_root.exists():
        raise FileNotFoundError(f"Image root not found: {images_root}")
    if not social_root.exists():
        raise FileNotFoundError(f"social root not found: {social_root}")

    exps = [args.exp] if args.exp != "all" else ["exp1", "exp2", "exp3", "exp4"]

    print(f"Images root: {images_root}")
    print(f"Data root:   {social_root}")
    print(f"Output dir:  {output_dir}")

    exp_level_data = {
        "exp1": _build_exp12_data(social_root, "exp1"),
        "exp2": _build_exp12_data(social_root, "exp2"),
        "exp3": _build_exp34_data(social_root, "exp3"),
        "exp4": _build_exp34_data(social_root, "exp4"),
    }

    if args.keep_charts:
        chart_root = output_dir / "barplot_cache"
        chart_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = TemporaryDirectory(prefix="barplot_cache_")
        chart_root = Path(temp_ctx.name)

    try:
        for exp in exps:
            exp_dir = images_root / exp
            images = sorted(exp_dir.glob("*.png")) if exp_dir.exists() else []
            images = [p for p in images if not _is_omitted_level(_level_from_image(p))]
            if args.max_slides is not None:
                images = images[: args.max_slides]

            slide_specs = _build_slide_specs(exp, images)
            out_path = output_dir / f"maps_observations_{exp}.pptx"

            print(f"\n{exp}: images={len(images)}, slides={len(slide_specs)}")
            _build_deck_for_exp(exp, slide_specs, exp_level_data[exp], out_path, chart_root / exp)
            print(f"saved: {out_path}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
