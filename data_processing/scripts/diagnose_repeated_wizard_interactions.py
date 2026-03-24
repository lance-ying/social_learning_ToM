#!/usr/bin/env python3
"""
Diagnostic scan for repeated wizard interactions in processed participant CSVs.

By default, this focuses on exp2 levels ending in "_2", since those are the
cases of interest for the current debugging thread.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan processed participant CSVs for repeated wizard interactions."
    )
    parser.add_argument(
        "--exp",
        default="exp2",
        help="Processed data experiment directory under data_processing/data_processed.",
    )
    parser.add_argument(
        "--level-suffix",
        default="_2",
        help="Only analyze levels whose ids end with this suffix. Use empty string for all levels.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many example participant-level runs to print.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the full diagnostic report as JSON.",
    )
    return parser.parse_args()


def parse_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_participant_interactions(csv_path: Path) -> dict[str, list[dict]]:
    per_level: dict[str, list[dict]] = defaultdict(list)
    current_level: str | None = None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("Level: "):
                current_level = line.split("Level: ", 1)[1].strip()
                per_level.setdefault(current_level, [])
                continue
            if current_level is None:
                continue
            if line.startswith("Timestamp,Type,"):
                continue
            if "," not in line:
                continue

            row = next(csv.reader([line]))
            row += [""] * (12 - len(row))
            event_type = row[1].strip()
            interaction_type = row[10].strip()

            if event_type != "INTERACTION" or interaction_type != "wizard_interaction":
                continue

            per_level[current_level].append(
                {
                    "timestamp": row[0].strip(),
                    "player_x": parse_float(row[3]),
                    "player_y": parse_float(row[4]),
                    "npc_x": parse_float(row[5]),
                    "npc_y": parse_float(row[6]),
                    "interaction_details": row[11].strip(),
                }
            )

    return per_level


def summarize_runs(data_dir: Path, level_suffix: str) -> dict:
    runs = []
    by_level: dict[str, list[dict]] = defaultdict(list)

    for csv_path in sorted(data_dir.glob("*.csv")):
        per_level = load_participant_interactions(csv_path)
        for level_id, interactions in per_level.items():
            if level_suffix and not level_id.endswith(level_suffix):
                continue

            coords = [
                (event["player_x"], event["player_y"])
                for event in interactions
                if event["player_x"] is not None and event["player_y"] is not None
            ]
            unique_coords = list(dict.fromkeys(coords))
            run = {
                "participant_file": csv_path.name,
                "level_id": level_id,
                "wizard_interaction_count": len(interactions),
                "unique_wizard_positions": len(unique_coords),
                "repeated_wizard_interactions": len(interactions) >= 2,
                "same_wizard_revisited": len(unique_coords) < len(coords),
                "interaction_details": [event["interaction_details"] for event in interactions],
                "positions": [[x, y] for (x, y) in coords],
                "events": interactions,
            }
            runs.append(run)
            by_level[level_id].append(run)

    level_summary = {}
    for level_id, level_runs in sorted(by_level.items()):
        counts = [run["wizard_interaction_count"] for run in level_runs]
        level_summary[level_id] = {
            "n_runs": len(level_runs),
            "mean_wizard_interaction_count": mean(counts) if counts else 0.0,
            "max_wizard_interaction_count": max(counts) if counts else 0,
            "runs_with_any_wizard_interaction": sum(run["wizard_interaction_count"] > 0 for run in level_runs),
            "runs_with_repeated_wizard_interactions": sum(run["repeated_wizard_interactions"] for run in level_runs),
            "runs_with_same_wizard_revisited": sum(run["same_wizard_revisited"] for run in level_runs),
        }

    repeated_runs = [run for run in runs if run["repeated_wizard_interactions"]]
    revisited_runs = [run for run in runs if run["same_wizard_revisited"]]

    return {
        "data_dir": str(data_dir),
        "level_suffix_filter": level_suffix,
        "n_runs": len(runs),
        "n_repeated_runs": len(repeated_runs),
        "n_same_wizard_revisited_runs": len(revisited_runs),
        "level_summary": level_summary,
        "repeated_runs": sorted(
            repeated_runs,
            key=lambda run: (
                run["wizard_interaction_count"],
                run["unique_wizard_positions"],
                run["level_id"],
                run["participant_file"],
            ),
            reverse=True,
        ),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data_processing" / "data_processed" / args.exp
    if not data_dir.exists():
        raise SystemExit(f"Missing processed data directory: {data_dir}")

    report = summarize_runs(data_dir, args.level_suffix)

    print(f"Data dir: {report['data_dir']}")
    print(f"Level suffix filter: {report['level_suffix_filter']!r}")
    print(f"Runs analyzed: {report['n_runs']}")
    print(f"Runs with repeated wizard interactions: {report['n_repeated_runs']}")
    print(f"Runs revisiting the same wizard position: {report['n_same_wizard_revisited_runs']}")
    print()
    print("Per-level summary:")
    for level_id, summary in report["level_summary"].items():
        print(
            f"  {level_id}: runs={summary['n_runs']}, "
            f"mean_interactions={summary['mean_wizard_interaction_count']:.2f}, "
            f"max_interactions={summary['max_wizard_interaction_count']}, "
            f"repeated_runs={summary['runs_with_repeated_wizard_interactions']}, "
            f"same_wizard_revisited={summary['runs_with_same_wizard_revisited']}"
        )

    if report["repeated_runs"]:
        print()
        print(f"Top {min(args.top_n, len(report['repeated_runs']))} repeated-interaction runs:")
        for run in report["repeated_runs"][: args.top_n]:
            print(
                f"  {run['level_id']} / {run['participant_file']}: "
                f"wizard_interactions={run['wizard_interaction_count']}, "
                f"unique_positions={run['unique_wizard_positions']}, "
                f"same_wizard_revisited={run['same_wizard_revisited']}, "
                f"details={run['interaction_details']}, "
                f"positions={run['positions']}"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print()
        print(f"Saved JSON -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
