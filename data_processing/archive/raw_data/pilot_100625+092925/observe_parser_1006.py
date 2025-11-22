
import re, csv
from pathlib import Path
import pandas as pd
import numpy as np
import json

LEVEL_RE = re.compile(r'^Level:\s*(.+?)\s*$', re.IGNORECASE)

def load_excluded_participants(summary_path: Path) -> set:
    """
    Load the summary CSV and return a set of filenames to exclude
    (participants with step counts below 100)
    """
    try:
        summary_df = pd.read_csv(summary_path)
        # Filter for participants with step counts below 0
        excluded = summary_df[summary_df['Total Steps'] < 0]['Filename'].tolist()
        # Convert to set for faster lookup and extract just the filename
        excluded_filenames = set()
        for filepath in excluded:
            filename = Path(filepath).name
            excluded_filenames.add(filename)
        print(f"Excluding {len(excluded_filenames)} participants with step counts below 0:")
        for filename in sorted(excluded_filenames):
            participant_steps = summary_df[summary_df['Filename'].str.contains(filename)]['Total Steps'].iloc[0]
            print(f"  {filename} ({participant_steps} steps)")
        return excluded_filenames
    except Exception as e:
        print(f"Warning: Could not load summary file {summary_path}: {e}")
        return set()

def parse_file_counts(path: Path) -> pd.DataFrame:
    """
    For a pilot export file:
      - Count OBSERVE events per level (sum across all activations within that file)
      - Count how many times each level "activates" (i.e., a new `Level: X` section with a following table)
    Returns a DataFrame with columns: file, level, observe_count, activation_count
    """
    observe_counts = {}
    activation_counts = {}

    current_level = None
    in_table = False
    seen_rows_in_this_activation = False  # to avoid incrementing twice for the same activation

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
                # reset section flags
                in_table = False
                seen_rows_in_this_activation = False
                # we *tentatively* set up counts dicts but only bump activation when we see a header
                observe_counts.setdefault(current_level, 0)
                activation_counts.setdefault(current_level, 0)
                continue

            # header indicates a (re)start of a table for the current level
            if current_level is not None and line.lower().startswith("timestamp,type"):
                in_table = True
                # Count an activation once per header occurrence for this level
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


def count_observes_and_means(paths, excluded_filenames=None):
    if excluded_filenames is None:
        excluded_filenames = set()
    
    frames = []
    processed_count = 0
    excluded_count = 0
    
    for p in paths:
        if p.name in excluded_filenames:
            excluded_count += 1
            print(f"Excluding file: {p.name}")
            continue
        
        frames.append(parse_file_counts(Path(p)))
        processed_count += 1
    
    print(f"Processed {processed_count} files, excluded {excluded_count} files")
    
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
    # Load excluded participants from summary file
    summary_path = Path("pilot_results_summary.csv")
    excluded_filenames = load_excluded_participants(summary_path)
    
    paths = Path("../../raw_data/pilot_100625+092925").glob("*.csv")
    per_file, overall, overall_dict, overall_json = count_observes_and_means(paths, excluded_filenames)
    print("per_file:")
    print(per_file)
    print("\noverall:")
    print(overall)
    print("\noverall_dict:")
    print(overall_dict)
    print("\noverall_json:")
    print(overall_json)