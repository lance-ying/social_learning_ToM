#!/usr/bin/env python3
"""
Observe parser that handles multiple NPCs (NPC 1 and NPC 2).
Adapted from observe_parser.py to handle 2 NPCs instead of 1.
"""
import re
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import List

LEVEL_RE = re.compile(r'^Level:\s*(.+?)\s*$', re.IGNORECASE)

def parse_file_counts(path: Path) -> pd.DataFrame:
    """
    For a CSV file:
      - Count OBSERVE events per level (sum across all activations within that file)
      - Count how many times each level "activates" (i.e., a new `Level: X` section with a following table)
    Returns a DataFrame with columns: file, level, observe_count, activation_count
    
    Now handles multiple NPC columns (NPC 1 X/Y, NPC 2 X/Y, or legacy NPC X/Y).
    """
    observe_counts = {}
    activation_counts = {}

    current_level = None
    in_table = False
    seen_rows_in_this_activation = False

    def bump_observe(level: str):
        observe_counts[level] = observe_counts.get(level, 0) + 1

    def bump_activation(level: str):
        activation_counts[level] = activation_counts.get(level, 0) + 1

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue

            m = LEVEL_RE.match(line)
            if m:
                current_level = m.group(1)
                in_table = False
                seen_rows_in_this_activation = False
                observe_counts.setdefault(current_level, 0)
                activation_counts.setdefault(current_level, 0)
                continue

            # header indicates a (re)start of a table for the current level
            if current_level is not None and line.lower().startswith("timestamp,type"):
                in_table = True
                bump_activation(current_level)
                seen_rows_in_this_activation = False
                continue

            if current_level is not None and in_table:
                # CSV-parse a single line
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = line.split(",")

                if len(row) >= 2:
                    type_field = (row[1] or "").strip().upper()
                    if "OBSERVE" in type_field:
                        bump_observe(current_level)

    # Build frame
    levels = sorted(set(observe_counts) | set(activation_counts))
    records = []
    for lvl in levels:
        records.append({
            "file": path.name,
            "level": lvl,
            "observe_count": int(observe_counts.get(lvl, 0) or 0),
            "activation_count": int(activation_counts.get(lvl, 0) or 0),
        })
    return pd.DataFrame.from_records(records, columns=["file","level","observe_count","activation_count"])


def count_observes_and_means(paths: List[Path]):
    """
    Process multiple CSV files and calculate statistics.
    Returns: per_file, overall, overall_dict, overall_json
    """
    frames = []
    for p in paths:
        frames.append(parse_file_counts(Path(p)))
    if not frames:
        per_file = pd.DataFrame(columns=["file","level","observe_count","activation_count","mean_observe_per_activation"])
        overall = pd.DataFrame(columns=["level","observe_count","activation_count","mean_observe_per_activation","bootstrap_sd","ci_lower","ci_upper"])
        overall_dict = {}
        overall_json = json.dumps(overall_dict)
    else:
        per_file = pd.concat(frames, ignore_index=True)
        # mean per file-level
        per_file["mean_observe_per_activation"] = per_file.apply(
            lambda r: (r["observe_count"] / r["activation_count"]) if r["activation_count"] else float("nan"),
            axis=1
        )
        # overall aggregate
        agg = per_file.groupby("level", as_index=False).agg(
            observe_count=("observe_count","sum"),
            activation_count=("activation_count","sum"),
        )
        agg["mean_observe_per_activation"] = agg.apply(
            lambda r: (r["observe_count"] / r["activation_count"]) if r["activation_count"] else float("nan"),
            axis=1
        )
        # bootstrap SD and 95% CI from per-file means (1000 resamples) for each level
        def _bootstrap_ci(values, n_boot=1000, alpha=0.05):
            values = np.asarray(values)
            values = values[~np.isnan(values)]
            if values.size == 0:
                return np.nan, np.nan, np.nan
            rng = np.random.default_rng()
            samples = rng.choice(values, size=(n_boot, values.size), replace=True)
            boot_means = samples.mean(axis=1)
            sd = float(boot_means.std(ddof=1))
            lower = float(np.quantile(boot_means, alpha/2))
            upper = float(np.quantile(boot_means, 1 - alpha/2))
            return sd, lower, upper

        per_level_means = per_file.groupby("level")["mean_observe_per_activation"].apply(list)

        boot_rows = (
            per_level_means.apply(lambda vals: _bootstrap_ci(vals, n_boot=1000))
            .apply(pd.Series)
            .rename(columns={0: "bootstrap_sd", 1: "ci_lower", 2: "ci_upper"})
            .reset_index()
        )

        overall = (
            agg.merge(boot_rows, on="level", how="left")
               .sort_values("mean_observe_per_activation", ascending=False)
        )
        # Build python dictionary keyed by level, and JSON string
        overall_dict = overall.set_index("level")[
            [
                "mean_observe_per_activation",
                "bootstrap_sd",
                "ci_lower",
                "ci_upper",
                "observe_count",
                "activation_count",
            ]
        ].to_dict(orient="index")
        overall_json = json.dumps(overall_dict, indent=2)
    return per_file, overall, overall_dict, overall_json

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/observe_parser_multi_npc.py <csv_directory>")
        sys.exit(1)
    
    csv_dir = Path(sys.argv[1])
    paths = list(csv_dir.glob("*.csv"))
    per_file, overall, overall_dict, overall_json = count_observes_and_means(paths)
    print("per_file:")
    print(per_file)
    print("\noverall:")
    print(overall)
    print("\noverall_dict:")
    print(overall_dict)

