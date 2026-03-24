#!/usr/bin/env python3
"""Summarize participant trajectories for a single processed level."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


MOVE_ACTIONS = {"up", "down", "left", "right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize human trajectories for one processed level."
    )
    parser.add_argument("--exp", required=True, help="Experiment id, e.g. exp2")
    parser.add_argument("--level", required=True, help="Level id, e.g. s521_2")
    parser.add_argument(
        "--output-json",
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=40,
        help="Number of condensed trace events to keep in representative previews.",
    )
    return parser.parse_args()


def iter_level_blocks(csv_path: Path):
    current_level = None
    block = []
    for line in csv_path.read_text().splitlines():
        if line.startswith("Level: "):
            if current_level is not None:
                yield current_level, block
            current_level = line.split(": ", 1)[1].strip()
            block = []
        elif current_level is not None:
            block.append(line)
    if current_level is not None:
        yield current_level, block


def parse_level_block(block: list[str], preview_limit: int) -> dict:
    move_steps = 0
    observe_steps = 0
    interaction_steps = 0
    wizard_interactions = []
    treasure_interactions = []
    condensed_trace = []
    prev_player_pos = None

    for raw in block:
        if raw.startswith("Timestamp,Type") or raw.startswith("timestamp,type"):
            continue
        row = next(csv.reader([raw]))
        if len(row) < 2:
            continue

        event_type = row[1]
        action = row[2] if len(row) > 2 else ""
        player_x = row[3] if len(row) > 3 else ""
        player_y = row[4] if len(row) > 4 else ""
        interaction_type = row[10] if len(row) > 10 else ""
        interaction_detail = row[11] if len(row) > 11 else ""
        current_player_pos = (player_x, player_y) if player_x and player_y else None

        if "AGENT_OBSERVE" in event_type or event_type == "OBSERVE":
            observe_steps += 1
            if len(condensed_trace) < preview_limit:
                condensed_trace.append(
                    {
                        "event": "observe",
                        "player_pos": [player_x, player_y],
                    }
                )
        elif event_type == "INTERACTION":
            interaction_steps += 1
            event = {
                "player_pos": [player_x, player_y],
                "interaction_type": interaction_type,
                "interaction_detail": interaction_detail,
            }
            if interaction_type == "wizard_interaction":
                wizard_interactions.append(event)
            elif interaction_type == "treasure_interaction":
                treasure_interactions.append(event)

            if len(condensed_trace) < preview_limit:
                condensed_trace.append(
                    {
                        "event": interaction_type or "interaction",
                        "player_pos": [player_x, player_y],
                        "detail": interaction_detail,
                    }
                )
        elif event_type == "GAME_ACTION" and action in MOVE_ACTIONS:
            if prev_player_pos is None or current_player_pos != prev_player_pos:
                move_steps += 1
            if len(condensed_trace) < preview_limit:
                condensed_trace.append(
                    {
                        "event": "move",
                        "action": action,
                        "player_pos": [player_x, player_y],
                    }
                )

        if current_player_pos is not None:
            prev_player_pos = current_player_pos

    move_cost = 3 * move_steps
    observe_cost = observe_steps
    interaction_cost = 5 * interaction_steps
    planning_cost = move_cost + interaction_cost
    total_cost = planning_cost + observe_cost

    return {
        "move_steps": move_steps,
        "observe_steps": observe_steps,
        "interaction_steps": interaction_steps,
        "move_cost": move_cost,
        "observe_cost": observe_cost,
        "interaction_cost": interaction_cost,
        "planning_cost": planning_cost,
        "total_cost": total_cost,
        "wizard_interactions": wizard_interactions,
        "treasure_interactions": treasure_interactions,
        "wizard_sequence_positions": [evt["player_pos"] for evt in wizard_interactions],
        "wizard_sequence_details": [evt["interaction_detail"] for evt in wizard_interactions],
        "condensed_trace": condensed_trace,
    }


def summarize_runs(rows: list[dict], preview_limit: int) -> dict:
    rows_sorted = sorted(rows, key=lambda row: (row["total_cost"], row["file"]))
    total_costs = [row["total_cost"] for row in rows_sorted]
    planning_costs = [row["planning_cost"] for row in rows_sorted]
    move_steps = [row["move_steps"] for row in rows_sorted]
    observe_steps = [row["observe_steps"] for row in rows_sorted]
    interaction_steps = [row["interaction_steps"] for row in rows_sorted]

    def representative(which: str) -> dict:
        if which == "min":
            row = rows_sorted[0]
        elif which == "median":
            row = rows_sorted[len(rows_sorted) // 2]
        else:
            row = rows_sorted[-1]
        return {
            "file": row["file"],
            "total_cost": row["total_cost"],
            "planning_cost": row["planning_cost"],
            "observe_cost": row["observe_cost"],
            "move_steps": row["move_steps"],
            "observe_steps": row["observe_steps"],
            "interaction_steps": row["interaction_steps"],
            "wizard_sequence_positions": row["wizard_sequence_positions"],
            "wizard_sequence_details": row["wizard_sequence_details"],
            "condensed_trace": row["condensed_trace"][:preview_limit],
        }

    seq_counter = Counter(
        tuple(tuple(pos) for pos in row["wizard_sequence_positions"]) for row in rows_sorted
    )
    detail_counter = Counter(tuple(row["wizard_sequence_details"]) for row in rows_sorted)

    return {
        "n_runs": len(rows_sorted),
        "mean_total_cost": statistics.mean(total_costs),
        "median_total_cost": statistics.median(total_costs),
        "mean_planning_cost": statistics.mean(planning_costs),
        "median_planning_cost": statistics.median(planning_costs),
        "mean_move_steps": statistics.mean(move_steps),
        "mean_observe_steps": statistics.mean(observe_steps),
        "mean_interaction_steps": statistics.mean(interaction_steps),
        "top_wizard_sequences_by_position": [
            {"count": count, "sequence": [list(pos) for pos in seq]}
            for seq, count in seq_counter.most_common(10)
        ],
        "top_wizard_sequences_by_detail": [
            {"count": count, "sequence": list(seq)}
            for seq, count in detail_counter.most_common(10)
        ],
        "representative_runs": {
            "min_cost": representative("min"),
            "median_cost": representative("median"),
            "max_cost": representative("max"),
        },
    }


def main() -> None:
    args = parse_args()
    data_dir = Path("data_processing/data_processed") / args.exp
    rows = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        for level_id, block in iter_level_blocks(csv_path):
            if level_id != args.level:
                continue
            parsed = parse_level_block(block, args.preview_limit)
            parsed["file"] = csv_path.name
            rows.append(parsed)

    if not rows:
        raise SystemExit(f"No runs found for {args.exp} / {args.level}")

    payload = {
        "exp": args.exp,
        "level": args.level,
        **summarize_runs(rows, args.preview_limit),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON -> {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
