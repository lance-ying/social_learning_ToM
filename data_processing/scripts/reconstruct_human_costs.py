#!/usr/bin/env python3
"""
Reconstruct human action costs from processed participant CSV files.

Outputs a JSON structure parallel to model reconstructed-cost files:
- summary
- per_case (aggregated by level)

Cost defaults follow the current repo-wide convention:
- move = 3
- interact = 5
- observe = 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


SKIP_LEVELS = {"comprehension_check", "experiment"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct human execution costs from processed CSV files."
    )
    parser.add_argument(
        "--exp",
        choices=["exp1", "exp2", "exp3", "exp4"],
        required=True,
        help="Experiment identifier.",
    )
    parser.add_argument(
        "--csv-dir",
        help="Directory containing processed participant CSVs. Defaults to data_processing/data_processed/<exp>.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to output JSON file.",
    )
    parser.add_argument("--move-cost", type=float, default=3.0)
    parser.add_argument("--interact-cost", type=float, default=5.0)
    parser.add_argument("--observe-cost", type=float, default=1.0)
    parser.add_argument(
        "--include-tutorials",
        action="store_true",
        help="Include tutorial levels instead of filtering them out.",
    )
    parser.add_argument(
        "--require-positive-total-steps-remaining",
        action="store_true",
        help="Only include participants whose summed LEVEL_COMPLETE 'Steps Remaining' across included levels is > 0.",
    )
    parser.add_argument(
        "--iqr-filter",
        action="store_true",
        help=(
            "Restrict to the IQR-cleaned participant set used by the plotting figures "
            "(plotting/common.py filtered_participant_paths). Writes a *_iqr_filtered.json by default."
        ),
    )
    return parser.parse_args()


def iqr_kept_stems(exp: str) -> set[str]:
    """Participant stems kept by the figures' IQR filter (plotting/common.py)."""
    repo_root = Path(__file__).resolve().parents[2]
    plotting_dir = repo_root / "plotting"
    if str(plotting_dir) not in sys.path:
        sys.path.insert(0, str(plotting_dir))
    from common import filtered_participant_paths

    return {path.stem for path in filtered_participant_paths(exp)}


def should_skip_level(exp: str, level: str, include_tutorials: bool) -> bool:
    if level in SKIP_LEVELS:
        return True
    if not include_tutorials:
        tutorial_prefixes = {
            "exp1": ("mod_s111", "mod_s112"),
            "exp2": ("s111_", "s112_"),
            "exp3": ("s111_", "s112_", "sm111_", "sm112_"),
            "exp4": ("s111_", "s112_", "sm111_", "sm112_"),
        }
        if any(level.startswith(prefix) for prefix in tutorial_prefixes.get(exp, ())):
            return True
    return False


def normalize_level(exp: str, level: str) -> str:
    level = level.strip()

    if exp == "exp1":
        if level.startswith("mod_"):
            if level.endswith("_1"):
                level = level[: -len("_1")]
            if not level.endswith("_ascii"):
                level = f"{level}_ascii"
        return level

    if exp == "exp2":
        return level

    if exp in {"exp3", "exp4"}:
        if "_scenario" in level:
            return level
        if level.startswith("sm"):
            parts = level.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return f"{parts[0]}_scenario{parts[1]}"
        return level

    return level


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def parse_participant_csv(csv_path: Path, exp: str, include_tutorials: bool) -> dict[str, dict[str, float]]:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_level = None
    in_table = False
    header = None
    type_idx = None
    steps_remaining_idx = None
    player_x_idx = None
    player_y_idx = None
    level_rows: dict[str, dict[str, float]] = {}
    prev_player_pos = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            in_table = False
            continue

        if line.startswith("Level:"):
            current_level = line.split(":", 1)[1].strip()
            in_table = False
            header = None
            type_idx = None
            steps_remaining_idx = None
            player_x_idx = None
            player_y_idx = None
            prev_player_pos = None
            if not should_skip_level(exp, current_level, include_tutorials):
                normalized = normalize_level(exp, current_level)
                level_rows.setdefault(
                    normalized,
                    {
                        "move_steps": 0.0,
                        "observe_steps": 0.0,
                        "interaction_steps": 0.0,
                        "final_steps_remaining": math.nan,
                    },
                )
            continue

        if current_level is None or should_skip_level(exp, current_level, include_tutorials):
            continue

        if "Timestamp" in line and "Type" in line:
            try:
                header = next(csv.reader([line]))
                header_lower = [h.strip().lower() for h in header]
                type_idx = header_lower.index("type") if "type" in header_lower else None
                player_x_idx = header_lower.index("player x") if "player x" in header_lower else None
                player_y_idx = header_lower.index("player y") if "player y" in header_lower else None
                steps_remaining_idx = (
                    header_lower.index("steps remaining")
                    if "steps remaining" in header_lower
                    else None
                )
                in_table = type_idx is not None
            except Exception:
                in_table = False
            continue

        if not in_table or header is None or type_idx is None:
            continue

        normalized = normalize_level(exp, current_level)
        if normalized not in level_rows:
            continue

        try:
            row = next(csv.reader([line]))
        except Exception:
            continue

        if len(row) <= type_idx:
            continue

        event_type = (row[type_idx] or "").strip().upper()
        level_data = level_rows[normalized]
        current_player_pos = None
        if (
            player_x_idx is not None
            and player_y_idx is not None
            and len(row) > max(player_x_idx, player_y_idx)
        ):
            x_text = (row[player_x_idx] or "").strip()
            y_text = (row[player_y_idx] or "").strip()
            if x_text and y_text:
                current_player_pos = (x_text, y_text)

        if event_type == "GAME_ACTION":
            if prev_player_pos is None or current_player_pos != prev_player_pos:
                level_data["move_steps"] += 1
        elif "AGENT_OBSERVE" in event_type or event_type == "OBSERVE":
            level_data["observe_steps"] += 1
        elif event_type == "INTERACTION":
            level_data["interaction_steps"] += 1
        elif event_type == "LEVEL_COMPLETE" and steps_remaining_idx is not None and len(row) > steps_remaining_idx:
            steps_text = (row[steps_remaining_idx] or "").strip()
            if steps_text:
                try:
                    level_data["final_steps_remaining"] = float(steps_text)
                except ValueError:
                    pass

        if current_player_pos is not None:
            prev_player_pos = current_player_pos

    # Drop levels with no actual actions recorded at all.
    return {
        level: stats
        for level, stats in level_rows.items()
        if (stats["move_steps"] + stats["observe_steps"] + stats["interaction_steps"]) > 0
    }


def participant_total_steps_remaining(levels: dict[str, dict[str, float]]) -> float:
    return sum(
        stats["final_steps_remaining"]
        for stats in levels.values()
        if not math.isnan(stats["final_steps_remaining"])
    )


def aggregate_levels(
    participant_level_costs: dict[str, dict[str, dict[str, float]]],
    move_cost: float,
    interact_cost: float,
    observe_cost: float,
) -> dict[str, dict[str, float]]:
    by_level: dict[str, list[dict[str, float]]] = defaultdict(list)

    for _participant, levels in participant_level_costs.items():
        for level, stats in levels.items():
            move_steps = stats["move_steps"]
            observe_steps = stats["observe_steps"]
            interaction_steps = stats["interaction_steps"]
            planning_steps = move_steps + interaction_steps
            planning_cost = move_steps * move_cost + interaction_steps * interact_cost
            observe_cost_val = observe_steps * observe_cost
            total_steps = planning_steps + observe_steps
            total_cost = planning_cost + observe_cost_val

            by_level[level].append(
                {
                    "move_steps": move_steps,
                    "observe_steps": observe_steps,
                    "interaction_steps": interaction_steps,
                    "planning_steps": planning_steps,
                    "total_steps": total_steps,
                    "move_cost": move_steps * move_cost,
                    "observe_cost": observe_cost_val,
                    "interaction_cost": interaction_steps * interact_cost,
                    "planning_cost": planning_cost,
                    "total_cost": total_cost,
                    "final_steps_remaining": stats["final_steps_remaining"],
                }
            )

    per_case = {}
    for level, entries in sorted(by_level.items()):
        move_steps = [e["move_steps"] for e in entries]
        observe_steps = [e["observe_steps"] for e in entries]
        interaction_steps = [e["interaction_steps"] for e in entries]
        planning_steps = [e["planning_steps"] for e in entries]
        total_steps = [e["total_steps"] for e in entries]
        move_costs = [e["move_cost"] for e in entries]
        observe_costs = [e["observe_cost"] for e in entries]
        interaction_costs = [e["interaction_cost"] for e in entries]
        planning_costs = [e["planning_cost"] for e in entries]
        total_costs = [e["total_cost"] for e in entries]
        steps_remaining = [
            e["final_steps_remaining"] for e in entries if not math.isnan(e["final_steps_remaining"])
        ]

        per_case[level] = {
            "n_participants": len(entries),
            "move_steps_mean": safe_mean(move_steps),
            "move_steps_median": safe_median(move_steps),
            "move_steps_sd": safe_sd(move_steps),
            "observe_steps_mean": safe_mean(observe_steps),
            "observe_steps_median": safe_median(observe_steps),
            "observe_steps_sd": safe_sd(observe_steps),
            "interaction_steps_mean": safe_mean(interaction_steps),
            "interaction_steps_median": safe_median(interaction_steps),
            "interaction_steps_sd": safe_sd(interaction_steps),
            "planning_steps_mean": safe_mean(planning_steps),
            "planning_steps_median": safe_median(planning_steps),
            "planning_steps_sd": safe_sd(planning_steps),
            "total_steps_mean": safe_mean(total_steps),
            "total_steps_median": safe_median(total_steps),
            "total_steps_sd": safe_sd(total_steps),
            "move_cost_mean": safe_mean(move_costs),
            "move_cost_median": safe_median(move_costs),
            "move_cost_sd": safe_sd(move_costs),
            "observe_cost_mean": safe_mean(observe_costs),
            "observe_cost_median": safe_median(observe_costs),
            "observe_cost_sd": safe_sd(observe_costs),
            "interaction_cost_mean": safe_mean(interaction_costs),
            "interaction_cost_median": safe_median(interaction_costs),
            "interaction_cost_sd": safe_sd(interaction_costs),
            "planning_cost_mean": safe_mean(planning_costs),
            "planning_cost_median": safe_median(planning_costs),
            "planning_cost_sd": safe_sd(planning_costs),
            "total_cost_mean": safe_mean(total_costs),
            "total_cost_median": safe_median(total_costs),
            "total_cost_sd": safe_sd(total_costs),
            "steps_remaining_mean": safe_mean(steps_remaining),
            "steps_remaining_median": safe_median(steps_remaining),
            "steps_remaining_sd": safe_sd(steps_remaining),
        }

    return per_case


def build_summary(per_case: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_case:
        return {
            "n_levels": 0,
            "mean_n_participants": 0.0,
            "mean_move_cost": 0.0,
            "median_move_cost": 0.0,
            "mean_observe_cost": 0.0,
            "median_observe_cost": 0.0,
            "mean_interaction_cost": 0.0,
            "median_interaction_cost": 0.0,
            "mean_planning_cost": 0.0,
            "median_planning_cost": 0.0,
            "mean_total_cost": 0.0,
            "median_total_cost": 0.0,
            "mean_steps_remaining": 0.0,
            "median_steps_remaining": 0.0,
        }

    cases = list(per_case.values())
    return {
        "n_levels": len(cases),
        "mean_n_participants": safe_mean([c["n_participants"] for c in cases]),
        "mean_move_cost": safe_mean([c["move_cost_mean"] for c in cases]),
        "median_move_cost": safe_median([c["move_cost_median"] for c in cases]),
        "mean_observe_cost": safe_mean([c["observe_cost_mean"] for c in cases]),
        "median_observe_cost": safe_median([c["observe_cost_median"] for c in cases]),
        "mean_interaction_cost": safe_mean([c["interaction_cost_mean"] for c in cases]),
        "median_interaction_cost": safe_median([c["interaction_cost_median"] for c in cases]),
        "mean_planning_cost": safe_mean([c["planning_cost_mean"] for c in cases]),
        "median_planning_cost": safe_median([c["planning_cost_median"] for c in cases]),
        "mean_total_cost": safe_mean([c["total_cost_mean"] for c in cases]),
        "median_total_cost": safe_median([c["total_cost_median"] for c in cases]),
        "mean_steps_remaining": safe_mean([c["steps_remaining_mean"] for c in cases]),
        "median_steps_remaining": safe_median([c["steps_remaining_median"] for c in cases]),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    csv_dir = Path(args.csv_dir) if args.csv_dir else repo_root / "data_processing" / "data_processed" / args.exp
    default_name = f"{args.exp}_human_costs_iqr_filtered.json" if args.iqr_filter else f"{args.exp}_human_costs.json"
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root / "data_processing" / "outputs" / "human_costs" / default_name
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    participant_level_costs = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        participant_level_costs[csv_path.stem] = parse_participant_csv(
            csv_path, exp=args.exp, include_tutorials=args.include_tutorials
        )

    filtered_out_participants: list[str] = []
    if args.iqr_filter:
        keep = iqr_kept_stems(args.exp)
        filtered = {}
        for participant_id, levels in participant_level_costs.items():
            if participant_id in keep:
                filtered[participant_id] = levels
            else:
                filtered_out_participants.append(participant_id)
        participant_level_costs = filtered
    elif args.require_positive_total_steps_remaining:
        filtered = {}
        for participant_id, levels in participant_level_costs.items():
            if participant_total_steps_remaining(levels) > 0:
                filtered[participant_id] = levels
            else:
                filtered_out_participants.append(participant_id)
        participant_level_costs = filtered

    per_case = aggregate_levels(
        participant_level_costs,
        move_cost=args.move_cost,
        interact_cost=args.interact_cost,
        observe_cost=args.observe_cost,
    )
    summary = build_summary(per_case)

    out = {
        "exp": args.exp,
        "csv_dir": str(csv_dir),
        "action_cost": {
            "move": args.move_cost,
            "interact": args.interact_cost,
            "observe": args.observe_cost,
        },
        "participant_filter": {
            "iqr_filter": args.iqr_filter,
            "require_positive_total_steps_remaining": args.require_positive_total_steps_remaining,
            "participants_kept": len(participant_level_costs),
            "participants_filtered_out": len(filtered_out_participants),
            "filtered_out_ids": filtered_out_participants,
        },
        "summary": summary,
        "per_case": per_case,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved human costs to: {output_file}")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
