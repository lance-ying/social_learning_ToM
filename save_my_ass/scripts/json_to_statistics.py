#!/usr/bin/env python3
"""
Unified pipeline for processing JSON files:
- Direct JSON → Statistics (fast path)
- Optional JSON → CSV generation (for inspection)
"""
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import numpy as np


# CSV generation helper functions
def ms_to_iso(ms):
    try:
        return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
    except Exception:
        return None


def pick_first(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def find_first_key(d: Dict[str, Any], pattern: str, default=None):
    rx = re.compile(pattern, re.I)
    for k, v in d.items():
        if rx.search(k):
            return v
    return default


def stringify(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def action_delta(action: str) -> Tuple[int, int]:
    a = (action or "").strip().lower()
    return {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }.get(a, (0, 0))


def build_report_files(json_path: str, out_dir: str) -> List[str]:
    """Generate CSV report files from JSON (only for users with prolificId)."""
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip Git LFS pointer files
    try:
        content = json_path.read_text(encoding="utf-8")
        if content.startswith("version https://git-lfs.github.com"):
            print(f"Skipping Git LFS pointer file: {json_path.name}")
            return []
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path.name}: {e}")
        return []
    except Exception as e:
        print(f"Error reading file {json_path.name}: {e}")
        return []
    
    users = data.get("users", {}) if isinstance(data, dict) else {}
    out_paths: List[str] = []

    for user_id, user_data in users.items():
        demographics = (user_data or {}).get("demographics") or {}
        
        # Only generate CSV for users with prolificId
        if not demographics.get("prolificId"):
            continue
        
        session_id = (user_data or {}).get("sessionId") or str(user_id)
        created_at = ms_to_iso((user_data or {}).get("startTime"))
        completed_at = ms_to_iso((user_data or {}).get("lastActivity"))

        # Build flat event rows
        rows = []
        levels = (user_data or {}).get("levels", {}) if isinstance(user_data, dict) else {}
        for level_id, level_data in levels.items():
            batches = (level_data or {}).get("batches", {})
            for batch_id, batch_data in (batches or {}).items():
                for idx, ev in enumerate((batch_data or {}).get("events", []) or []):
                    ts_ms = ev.get("timestamp")
                    etype = (ev.get("type") or "").upper()
                    payload = ev.get("data", {})
                    if not isinstance(payload, dict):
                        payload = {"_raw": payload}
                    action = payload.get("action")
                    player_pos = payload.get("playerPos") or {}
                    agent_pos  = payload.get("agentPos")  or {}
                    agent_id = payload.get("agentId")  # Get agent ID (1 or 2)

                    steps_remaining = pick_first(payload, "stepsRemaining","remainingSteps","stepsLeft","steps_remaining","steps","StepsRemaining")
                    if steps_remaining is None:
                        steps_remaining = find_first_key(payload, r"(step).*(remain)|remaining")

                    path_type = pick_first(payload, "pathType","npcPathType","agentPathType")
                    if path_type is None:
                        path_type = find_first_key(payload, r"path.*type|npc.*path.*type")

                    path_index = pick_first(payload, "pathIndex","currentPathIndex","npcPathIndex")
                    if path_index is None:
                        path_index = find_first_key(payload, r"path.*index")

                    interaction_type = pick_first(payload, "interactionType","interaction","typeOfInteraction")
                    if interaction_type is None and "INTERACTION" in etype:
                        interaction_type = etype.lower()

                    detail_keys = ["detail","details","text","message","tooltip","tooltipText","dialog","reward","item","result","color","amuletColor","spell","wizard","treasure"]
                    dvals = []
                    for k, v in payload.items():
                        if any(tok in k.lower() for tok in detail_keys):
                            if not isinstance(v, (dict, list)) and v not in (None, ""):
                                dvals.append(str(v))
                    interaction_details = ", ".join(dict.fromkeys(dvals)) if dvals else ""

                    # Initialize NPC 1 and NPC 2 positions (default to None)
                    npc1_x = npc1_y = npc2_x = npc2_y = None
                    npc1_path_type = npc1_path_index = npc2_path_type = npc2_path_index = None
                    
                    # Assign to NPC 1 or NPC 2 based on agentId
                    if agent_id == 1:
                        npc1_x = agent_pos.get("x")
                        npc1_y = agent_pos.get("y")
                        npc1_path_type = path_type
                        npc1_path_index = path_index
                    elif agent_id == 2:
                        npc2_x = agent_pos.get("x")
                        npc2_y = agent_pos.get("y")
                        npc2_path_type = path_type
                        npc2_path_index = path_index
                    # If no agentId specified, default to NPC 1 (backward compatibility)
                    elif agent_pos.get("x") is not None or agent_pos.get("y") is not None:
                        npc1_x = agent_pos.get("x")
                        npc1_y = agent_pos.get("y")
                        npc1_path_type = path_type
                        npc1_path_index = path_index

                    rows.append({
                        "level_id": level_id,
                        "timestamp_ms": ts_ms,
                        "Timestamp": ms_to_iso(ts_ms) or "",
                        "Type": "GAME_ACTION" if action not in (None, "") else etype,
                        "Player Action": action or "",
                        "Player X": player_pos.get("x"),
                        "Player Y": player_pos.get("y"),
                        "NPC 1 X": npc1_x,
                        "NPC 1 Y": npc1_y,
                        "NPC 2 X": npc2_x,
                        "NPC 2 Y": npc2_y,
                        "Steps Remaining": steps_remaining,
                        "NPC 1 Path Type": npc1_path_type,
                        "NPC 1 Path Index": npc1_path_index,
                        "NPC 2 Path Type": npc2_path_type,
                        "NPC 2 Path Index": npc2_path_index,
                        "Interaction Type": stringify(interaction_type),
                        "Interaction Details": stringify(interaction_details),
                    })

        # Comprehension check
        passed = False
        for level_id, level_data in (levels or {}).items():
            for batch_id, batch_data in (level_data.get("batches") or {}).items():
                for ev in (batch_data or {}).get("events", []) or []:
                    pdata = ev.get("data") or {}
                    if isinstance(pdata, dict) and pdata.get("allAnswersCorrect") is True:
                        passed = True
                        break

        # Fallback created/completed from events
        if rows and not created_at:
            created_at = ms_to_iso(min(r["timestamp_ms"] for r in rows if r["timestamp_ms"] is not None))
        if rows and not completed_at:
            completed_at = ms_to_iso(max(r["timestamp_ms"] for r in rows if r["timestamp_ms"] is not None))

        # CSV filename is just the user ID since each JSON has its own directory
        out_path = Path(out_dir) / f"{str(user_id).replace('/', '_')}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Session ID", session_id])
            w.writerow(["Completed At", completed_at or ""])
            w.writerow(["Created At", created_at or ""])
            w.writerow([])
            w.writerow(["User Data"])
            if demographics:
                for key in ["prolificId","age","gender","feedback"]:
                    if key in demographics:
                        w.writerow([key, stringify(demographics[key])])
                for k in sorted(k for k in demographics.keys() if k not in {"prolificId","age","gender","feedback"}):
                    w.writerow([k, stringify(demographics[k])])
            else:
                w.writerow(["(no demographics found)",""])
            w.writerow([])
            w.writerow(["Comprehension Check"])
            w.writerow(["Passed", "TRUE" if passed else "FALSE"])
            w.writerow([])

            if rows:
                df = pd.DataFrame(rows)
                for level_id, gl in df.groupby("level_id", dropna=False):
                    w.writerow([f"Level: {level_id}"])
                    cols = ["Timestamp","Type","Player Action","Player X","Player Y","NPC 1 X","NPC 1 Y","NPC 2 X","NPC 2 Y","Steps Remaining","NPC 1 Path Type","NPC 1 Path Index","NPC 2 Path Type","NPC 2 Path Index","Interaction Type","Interaction Details"]
                    gl = gl.sort_values(by=["timestamp_ms"]).reset_index(drop=True)

                    # Compute post-move positions for action rows when missing
                    last_px = last_py = None
                    last_npc1_x = last_npc1_y = last_npc2_x = last_npc2_y = None
                    for i in range(len(gl)):
                        px, py = gl.at[i, "Player X"], gl.at[i, "Player Y"]
                        npc1_x, npc1_y = gl.at[i, "NPC 1 X"], gl.at[i, "NPC 1 Y"]
                        npc2_x, npc2_y = gl.at[i, "NPC 2 X"], gl.at[i, "NPC 2 Y"]
                        
                        if pd.notna(px): last_px = float(px)
                        if pd.notna(py): last_py = float(py)
                        if pd.notna(npc1_x): last_npc1_x = float(npc1_x)
                        if pd.notna(npc1_y): last_npc1_y = float(npc1_y)
                        if pd.notna(npc2_x): last_npc2_x = float(npc2_x)
                        if pd.notna(npc2_y): last_npc2_y = float(npc2_y)

                        if gl.at[i, "Type"] == "GAME_ACTION" and (pd.isna(px) or pd.isna(py)):
                            if last_px is not None and last_py is not None:
                                dx, dy = action_delta(gl.at[i, "Player Action"])
                                nx, ny = last_px + dx, last_py + dy
                                gl.at[i, "Player X"] = nx
                                gl.at[i, "Player Y"] = ny
                                last_px, last_py = nx, ny

                    # Forward-fill remaining (not Timestamp/Player Action)
                    ffill_cols = ["Type","Player X","Player Y","NPC 1 X","NPC 1 Y","NPC 2 X","NPC 2 Y","Steps Remaining","NPC 1 Path Type","NPC 1 Path Index","NPC 2 Path Type","NPC 2 Path Index"]
                    for c in ffill_cols:
                        if c in gl.columns:
                            gl[c] = gl[c].replace({"": pd.NA, "nan": pd.NA})
                    gl[ffill_cols] = gl[ffill_cols].ffill()

                    w.writerow(cols)
                    for _, r in gl.iterrows():
                        w.writerow([r.get(c, "") if pd.notna(r.get(c, "")) else "" for c in cols])
                    w.writerow([])

        out_paths.append(str(out_path))

    return out_paths


def parse_json_for_statistics(json_path: Path, only_prolific: bool = True) -> pd.DataFrame:
    """
    Directly parse JSON to extract statistics without generating CSV.
    Returns DataFrame with columns: file, level, observe_count, activation_count
    
    Args:
        json_path: Path to JSON file
        only_prolific: If True, only count statistics from users with prolificId
    """
    observe_counts = defaultdict(int)
    activation_counts = defaultdict(int)
    
    # Skip Git LFS pointer files
    try:
        content = json_path.read_text(encoding="utf-8")
        if content.startswith("version https://git-lfs.github.com"):
            print(f"Skipping Git LFS pointer file: {json_path.name}")
            return pd.DataFrame(columns=["file", "level", "observe_count", "activation_count"])
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path.name}: {e}")
        return pd.DataFrame(columns=["file", "level", "observe_count", "activation_count"])
    except Exception as e:
        print(f"Error reading file {json_path.name}: {e}")
        return pd.DataFrame(columns=["file", "level", "observe_count", "activation_count"])
    
    users = data.get("users", {}) if isinstance(data, dict) else {}
    
    for user_id, user_data in users.items():
        # Filter by prolific ID if requested
        if only_prolific:
            demographics = (user_data or {}).get("demographics") or {}
            if not demographics.get("prolificId"):
                continue  # Skip users without prolificId
        
        levels = (user_data or {}).get("levels", {}) if isinstance(user_data, dict) else {}
        
        for level_id, level_data in levels.items():
            batches = (level_data or {}).get("batches", {})
            
            # Count activations (each batch is an activation)
            activation_count = len(batches)
            if activation_count > 0:
                activation_counts[level_id] += activation_count
            
            # Count OBSERVE events across all batches
            for batch_id, batch_data in (batches or {}).items():
                events = (batch_data or {}).get("events", []) or []
                for ev in events:
                    etype = (ev.get("type") or "").upper()
                    if "OBSERVE" in etype:
                        observe_counts[level_id] += 1
    
    # Build DataFrame
    all_levels = set(observe_counts.keys()) | set(activation_counts.keys())
    records = []
    for level in sorted(all_levels):
        records.append({
            "file": json_path.name,
            "level": level,
            "observe_count": int(observe_counts.get(level, 0)),
            "activation_count": int(activation_counts.get(level, 0)),
        })
    
    return pd.DataFrame.from_records(records, columns=["file", "level", "observe_count", "activation_count"])


def has_prolific_id(json_path: Path) -> bool:
    """Check if a JSON file contains at least one user with a prolificId."""
    try:
        content = json_path.read_text(encoding="utf-8")
        if content.startswith("version https://git-lfs.github.com"):
            return False
        data = json.loads(content)
        users = data.get("users", {}) if isinstance(data, dict) else {}
        
        for user_data in users.values():
            demographics = (user_data or {}).get("demographics") or {}
            if demographics.get("prolificId"):
                return True
        return False
    except Exception:
        return False


def process_json_files_for_statistics(
    json_paths: List[Path],
    generate_csv: bool = False,
    csv_output_dir: Optional[Path] = None,
    csv_only_prolific: bool = True,
    stats_only_prolific: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, str]:
    """
    Process multiple JSON files directly to statistics.
    
    Args:
        json_paths: List of JSON file paths
        generate_csv: If True, also generate CSV files (slower but useful for inspection)
        csv_output_dir: Directory for CSV output (only used if generate_csv=True)
        csv_only_prolific: If True, only generate CSVs for users with prolificId
        stats_only_prolific: If True, only count statistics from users with prolificId
    
    Returns:
        Same as observe_parser_1007.count_observes_and_means:
        (per_file_df, overall_df, results_dict, results_json)
    """
    frames = []
    processed_count = 0
    csv_generated_count = 0
    csv_skipped_count = 0
    
    for json_path in json_paths:
        # Parse JSON directly for statistics (filter by prolific ID if requested)
        df = parse_json_for_statistics(json_path, only_prolific=stats_only_prolific)
        frames.append(df)
        processed_count += 1
        
        # Optionally generate CSV files (only for prolific IDs if requested)
        if generate_csv and csv_output_dir:
            should_generate = True
            if csv_only_prolific:
                should_generate = has_prolific_id(json_path)
                if not should_generate:
                    csv_skipped_count += 1
            
            if should_generate:
                csv_output_dir.mkdir(parents=True, exist_ok=True)
                # Create subdirectory for this JSON file
                json_subdir = csv_output_dir / json_path.stem
                build_report_files(str(json_path), str(json_subdir))
                csv_generated_count += 1
    
    print(f"Processed {processed_count} JSON files")
    if generate_csv:
        print(f"  Generated {csv_generated_count} CSV files")
        if csv_only_prolific and csv_skipped_count > 0:
            print(f"  Skipped {csv_skipped_count} files without prolificId")
    
    if not frames:
        per_file = pd.DataFrame(columns=["file", "level", "observe_count", "activation_count", "mean_observe_per_activation"])
        overall = pd.DataFrame(columns=["level", "observe_count", "activation_count", "mean_observe_per_activation", "bootstrap_sd", "ci_lower", "ci_upper"])
        overall_dict = {}
        overall_json = json.dumps(overall_dict)
    else:
        per_file = pd.concat(frames, ignore_index=True)
        
        # Calculate mean per file-level
        per_file["mean_observe_per_activation"] = per_file.apply(
            lambda r: (r["observe_count"] / r["activation_count"]) if r["activation_count"] > 0 else float("nan"),
            axis=1
        )
        
        # Overall aggregate
        agg = per_file.groupby("level", as_index=False).agg(
            observe_count=("observe_count", "sum"),
            activation_count=("activation_count", "sum"),
        )
        agg["mean_observe_per_activation"] = agg.apply(
            lambda r: (r["observe_count"] / r["activation_count"]) if r["activation_count"] > 0 else float("nan"),
            axis=1
        )
        
        # Bootstrap SD and 95% CI from per-file means (1000 resamples) for each level
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
        
        # Build python dictionary keyed by level
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
    
    # Example usage
    if len(sys.argv) > 1:
        json_dir = Path(sys.argv[1])
        generate_csv = "--csv" in sys.argv
        csv_output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and generate_csv else None
    else:
        json_dir = Path("data_raw")
        generate_csv = False
        csv_output_dir = None
    
    json_files = sorted(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    per_file, overall, dict_result, json_result = process_json_files_for_statistics(
        json_files,
        generate_csv=generate_csv,
        csv_output_dir=csv_output_dir
    )
    
    print("\nResults:")
    print(json_result)

