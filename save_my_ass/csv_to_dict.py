
import re, csv
from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import defaultdict

LEVEL_RE = re.compile(r'^Level:\s*(.+?)\s*$', re.IGNORECASE)

def parse_agent_observes(path: Path) -> dict:
    """
    Parse CSV file to extract AGENT_OBSERVE events with agent IDs and activation counts.
    Returns a dict: { level_name: { "observations": [...], "agent2_count": X, "agent3_count": Y, 
                                    "t": total, "activation_count": N, "mean_observe_per_activation": t/N } }
    """
    level_observations = defaultdict(list)  # level -> list of agent observations in order
    level_activations = defaultdict(int)  # level -> activation count
    
    current_level = None
    in_table = False
    header_row = None
    observed_agent_idx = None
    
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
                header_row = None
                observed_agent_idx = None
                continue

            # Find header row to get column index for "Observed Agent ID"
            # Also count this as an activation
            if current_level is not None and line.lower().startswith("timestamp,type"):
                in_table = True
                header_row = line
                # Count activation when we see a header
                if current_level and current_level not in ["comprehension_check", "experiment", "s111_1"]:
                    level_activations[current_level] += 1
                
                # Parse header to find "Observed Agent ID" column index
                try:
                    headers = next(csv.reader([line]))
                    observed_agent_idx = None
                    for idx, header in enumerate(headers):
                        if "observed agent id" in header.lower():
                            observed_agent_idx = idx
                            break
                except Exception:
                    pass
                continue

            if current_level is not None and in_table and observed_agent_idx is not None:
                # CSV-parse a single line
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = line.split(",")

                if len(row) > observed_agent_idx:
                    type_field = (row[1] if len(row) > 1 else "").strip().upper()
                    if "AGENT_OBSERVE" in type_field:
                        # Extract observed agent ID
                        agent_id_str = row[observed_agent_idx].strip() if len(row) > observed_agent_idx else ""
                        try:
                            agent_id = float(agent_id_str)
                            if agent_id == 2.0:
                                level_observations[current_level].append("agent2")
                            elif agent_id == 3.0:
                                level_observations[current_level].append("agent3")
                            # Could add more agents if needed
                        except (ValueError, TypeError):
                            pass  # Skip if can't parse agent ID

    # Build result dict matching the JSON structure, with activation counts
    # Include ALL levels that had activations, even if they have 0 observations
    all_levels = set(level_observations.keys()) | set(level_activations.keys())
    result = {}
    
    for level in all_levels:
        # Skip non-game levels
        if level in ["comprehension_check", "experiment", "s111_1"]:
            continue
        
        observations = level_observations.get(level, [])
        agent2_count = sum(1 for obs in observations if obs == "agent2")
        agent3_count = sum(1 for obs in observations if obs == "agent3")
        total = len(observations)
        activation_count = level_activations.get(level, 1)  # Default to 1 if no activations found
        
        # Calculate mean observations per activation
        mean_observe_per_activation = total / activation_count if activation_count > 0 else 0.0
        
        result[level] = {
            "observations": observations,
            "agent2_count": agent2_count,
            "agent3_count": agent3_count,
            "t": total,
            "activation_count": activation_count,
            "mean_observe_per_activation": mean_observe_per_activation
        }
    
    return result

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


def count_observes_and_means(paths):
    frames = []
    processed_count = 0

    for p in paths:
        frames.append(parse_file_counts(Path(p)))
        processed_count += 1

    print(f"Processed {processed_count} files")
    
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

def aggregate_agent_observes(csv_dir: Path) -> dict:
    """
    Process all CSV files in a directory and aggregate agent observations.
    Returns a dict with mean_observe_per_activation calculated across all files.
    """
    all_level_data = defaultdict(lambda: {
        "observations": [], 
        "agent2_count": 0, 
        "agent3_count": 0, 
        "t": 0,
        "activation_count": 0
    })
    
    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"Processing {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        file_data = parse_agent_observes(csv_file)
        # Aggregate observations and activations across all files for each level
        for level, data in file_data.items():
            all_level_data[level]["observations"].extend(data["observations"])
            all_level_data[level]["agent2_count"] += data["agent2_count"]
            all_level_data[level]["agent3_count"] += data["agent3_count"]
            all_level_data[level]["t"] += data["t"]
            all_level_data[level]["activation_count"] += data["activation_count"]
    
    # Convert to final format with mean_observe_per_activation
    result = {}
    for level, data in all_level_data.items():
        activation_count = data["activation_count"]
        total_observations = data["t"]
        mean_observe_per_activation = total_observations / activation_count if activation_count > 0 else 0.0
        
        result[level] = {
            "observations": data["observations"],
            "agent2_count": data["agent2_count"],
            "agent3_count": data["agent3_count"],
            "t": total_observations,
            "activation_count": activation_count,
            "mean_observe_per_activation": mean_observe_per_activation
        }
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_dir = Path(sys.argv[1])
    else:
        csv_dir = Path("test_output")
    
    print("\n" + "="*60)
    print(f"PROCESSING AGENT OBSERVES: {csv_dir}")
    print("="*60)
    
    # Parse agent observations
    agent_observe_dict = aggregate_agent_observes(csv_dir)
    
    print(f"\nFound {len(agent_observe_dict)} levels with agent observations")
    print("\nSample results:")
    for level, data in list(agent_observe_dict.items())[:5]:
        print(f"  {level}: {data['t']} total observations ({data['agent2_count']} agent2, {data['agent3_count']} agent3)")
    
    # Save to JSON file for comparison
    output_file = Path("agent_observes_results.json")
    with open(output_file, "w") as f:
        json.dump(agent_observe_dict, f, indent=2)
    
    print(f"\n✓ Saved results to: {output_file}")
    
    # Also save as Python dict
    output_py = Path("agent_observes_results.py")
    with open(output_py, "w") as f:
        f.write("# Generated agent observation statistics\n")
        f.write(f"agent_observes_dict = {repr(agent_observe_dict)}\n")
    
    print(f"✓ Saved Python dict to: {output_py}")