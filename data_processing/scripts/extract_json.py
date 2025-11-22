# json_to_report_ffill_movecalc.py
import csv, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd

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
                    agent_id = payload.get("agentId")  # Get agent ID (1, 2, 3, etc.)
                    
                    # For agent_observe events, track which agent was observed
                    observed_agent_id = None
                    if "AGENT_OBSERVE" in etype or "agent_observe" in etype.lower():
                        observed_agent_id = agent_id

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
                    # Note: agentId can be 1, 2, 3, etc., but we only track NPC 1 and NPC 2 positions
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
                        "Observed Agent ID": observed_agent_id,
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
                    cols = ["Timestamp","Type","Player Action","Player X","Player Y","Observed Agent ID","NPC 1 X","NPC 1 Y","NPC 2 X","NPC 2 Y","Steps Remaining","NPC 1 Path Type","NPC 1 Path Index","NPC 2 Path Type","NPC 2 Path Index","Interaction Type","Interaction Details"]
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

                    # Forward-fill remaining (not Timestamp/Player Action/Observed Agent ID)
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

def process_directory(json_dir: str, out_dir: str = None) -> List[str]:
    """Process all JSON files in a directory and generate CSV reports."""
    json_dir = Path(json_dir)
    if out_dir is None:
        out_dir = json_dir / "csv_reports"
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_outputs = []
    json_files = sorted(json_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        # Create a subdirectory for each JSON file
        json_subdir = out_dir / json_file.stem
        outputs = build_report_files(str(json_file), str(json_subdir))
        all_outputs.extend(outputs)
        if outputs:
            print(f"  Generated {len(outputs)} CSV file(s) in {json_subdir.name}/")
        else:
            print(f"  No CSV files generated (may be empty or invalid)")
    
    print(f"\nTotal CSV files generated: {len(all_outputs)}")
    return all_outputs

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        out_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Check if input is a file or directory
        if input_path.is_file() and input_path.suffix == ".json":
            # Single JSON file
            if out_dir is None:
                out_dir = input_path.stem + "_csvs"
            print(f"Processing single JSON file: {input_path.name}")
            outputs = build_report_files(str(input_path), str(out_dir))
            print(f"Generated {len(outputs)} CSV file(s) in {out_dir}/")
        elif input_path.is_dir():
            # Directory of JSON files
            process_directory(str(input_path), out_dir)
        else:
            print(f"Error: {input_path} is not a valid JSON file or directory")
            sys.exit(1)
    else:
        # Default to processing the restrcuture_downloaded_data directory
        json_dir = "restrcuture_downloaded_data"
        out_dir = None
        process_directory(json_dir, out_dir)
