#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EXP=""
MODEL_LABEL=""
STEPS_FILE=""
REPLAY_TRACE_FILE=""
INFERENCE_FILE=""
PROBLEM_DIR=""
OUTPUT_FILE=""
HUMAN_COSTS_FILE=""
JOBS="${JOBS:-4}"
MOVE_COST="${MOVE_COST:-3}"
INTERACT_COST="${INTERACT_COST:-5}"
OBSERVE_COST="${OBSERVE_COST:-1}"
POSTERIOR_CANDIDATE_RULE="${POSTERIOR_CANDIDATE_RULE:-prob_threshold}"
POSTERIOR_MASS_THRESHOLD="${POSTERIOR_MASS_THRESHOLD:-0.9}"
POSTERIOR_PROB_THRESHOLD="${POSTERIOR_PROB_THRESHOLD:-0.1}"
KEEP_TEMP="${KEEP_TEMP:-0}"
DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING="${DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_reconstruct_sharded.sh \
    --exp exp1|exp2|exp3|exp4 \
    --model <label> \
    --steps-file <path> \
    [--replay-trace-file <path>] \
    --inference-file <path> \
    --human-costs-file <path> \
    --problem-dir <path> \
    --output-file <path> \
    --jobs <n> \
    [--disable-exp4-interaction-outcome-pruning] \
    [--move-cost 3] [--interact-cost 5] [--observe-cost 1] \
    [--posterior-candidate-rule prob_threshold|top_mass|positive_support] \
    [--posterior-mass-threshold 0.9]
    [--posterior-prob-threshold 0.1]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)
      EXP="$2"
      shift 2
      ;;
    --model)
      MODEL_LABEL="$2"
      shift 2
      ;;
    --steps-file)
      STEPS_FILE="$2"
      shift 2
      ;;
    --replay-trace-file)
      REPLAY_TRACE_FILE="$2"
      shift 2
      ;;
    --inference-file)
      INFERENCE_FILE="$2"
      shift 2
      ;;
    --human-costs-file)
      HUMAN_COSTS_FILE="$2"
      shift 2
      ;;
    --problem-dir)
      PROBLEM_DIR="$2"
      shift 2
      ;;
    --output-file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --disable-exp4-interaction-outcome-pruning)
      DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING=1
      shift
      ;;
    --move-cost)
      MOVE_COST="$2"
      shift 2
      ;;
    --interact-cost)
      INTERACT_COST="$2"
      shift 2
      ;;
    --observe-cost)
      OBSERVE_COST="$2"
      shift 2
      ;;
    --posterior-candidate-rule)
      POSTERIOR_CANDIDATE_RULE="$2"
      shift 2
      ;;
    --posterior-mass-threshold)
      POSTERIOR_MASS_THRESHOLD="$2"
      shift 2
      ;;
    --posterior-prob-threshold)
      POSTERIOR_PROB_THRESHOLD="$2"
      shift 2
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$EXP" || -z "$MODEL_LABEL" || -z "$STEPS_FILE" || -z "$PROBLEM_DIR" || -z "$OUTPUT_FILE" ]]; then
  usage >&2
  exit 1
fi

if [[ "$EXP" != "exp1" && "$EXP" != "exp2" && "$EXP" != "exp3" && "$EXP" != "exp4" ]]; then
  echo "Sharded runner only supports exp1, exp2, exp3, or exp4, got: $EXP" >&2
  exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be a positive integer, got: $JOBS" >&2
  exit 1
fi

if [[ ! -f "$STEPS_FILE" ]]; then
  echo "Missing steps file: $STEPS_FILE" >&2
  exit 1
fi

if [[ -n "$REPLAY_TRACE_FILE" && ! -f "$REPLAY_TRACE_FILE" ]]; then
  echo "Missing replay trace file: $REPLAY_TRACE_FILE" >&2
  exit 1
fi

if [[ ! -d "$PROBLEM_DIR" ]]; then
  echo "Missing problem dir: $PROBLEM_DIR" >&2
  exit 1
fi

if [[ -z "$HUMAN_COSTS_FILE" || ! -f "$HUMAN_COSTS_FILE" ]]; then
  echo "Missing human costs file: $HUMAN_COSTS_FILE" >&2
  exit 1
fi

REPLAY_TRACE_ARGS=()
if [[ -n "$REPLAY_TRACE_FILE" ]]; then
  REPLAY_TRACE_ARGS=(--replay-trace-file "$REPLAY_TRACE_FILE")
fi

declare -a EXP4_DISABLE_ARGS=()
if [[ "$EXP" == "exp4" && "$MODEL_LABEL" != "full_model" && "$MODEL_LABEL" != "social_mentalizing" && "$DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING" == "1" ]]; then
  EXP4_DISABLE_ARGS=(--disable-exp4-interaction-outcome-pruning)
fi

if [[ "$JOBS" -eq 1 ]]; then
  cmd=(
    julia --project=. scripts/utilities/reconstruct_model_costs.jl
    --exp "$EXP"
    --model "$MODEL_LABEL"
    --steps-file "$STEPS_FILE"
    "${REPLAY_TRACE_ARGS[@]}"
    --inference-file "$INFERENCE_FILE"
    --restrict-to-human-levels
    --human-costs-file "$HUMAN_COSTS_FILE"
    --problem-dir "$PROBLEM_DIR"
    --move-cost "$MOVE_COST"
    --interact-cost "$INTERACT_COST"
    --observe-cost "$OBSERVE_COST"
    --posterior-candidate-rule "$POSTERIOR_CANDIDATE_RULE"
    --posterior-mass-threshold "$POSTERIOR_MASS_THRESHOLD"
    --posterior-prob-threshold "$POSTERIOR_PROB_THRESHOLD"
    --output-file "$OUTPUT_FILE"
  )
  if [[ ${#EXP4_DISABLE_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXP4_DISABLE_ARGS[@]}")
  fi
  exec "${cmd[@]}"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/reconstruct_shards.XXXXXX")"
cleanup() {
  if [[ "$KEEP_TEMP" != "1" ]]; then
    rm -rf "$TMP_DIR"
  else
    echo "Keeping shard temp dir: $TMP_DIR"
  fi
}
trap cleanup EXIT

python3 - "$STEPS_FILE" "$TMP_DIR" "$JOBS" "$EXP" <<'PY'
import json
import os
import sys
from collections import OrderedDict

steps_file, tmp_dir, jobs_raw, exp = sys.argv[1:5]
jobs = int(jobs_raw)

with open(steps_file) as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

def group_name(key: str) -> str:
    if exp in ("exp3", "exp4") and "_scenario" in key:
        return key.split("_scenario", 1)[0]
    if exp in ("exp1", "exp2") and "_" in key:
        return key.split("_", 1)[0]
    return key

groups = OrderedDict()
for key, value in data.items():
    g = group_name(key)
    if g not in groups:
        groups[g] = OrderedDict()
    groups[g][key] = value

n_shards = min(jobs, max(len(groups), 1))
shards = [OrderedDict() for _ in range(n_shards)]
loads = [0 for _ in range(n_shards)]

for _, entries in groups.items():
    idx = min(range(n_shards), key=lambda i: (loads[i], i))
    shards[idx].update(entries)
    loads[idx] += len(entries)

manifest = []
for i, shard in enumerate(shards):
    if not shard:
        continue
    path = os.path.join(tmp_dir, f"shard_{i:02d}.json")
    with open(path, "w") as f:
        json.dump(shard, f, indent=2)
    manifest.append(path)
    print(f"Shard {i + 1}: {len(shard)} cases -> {path}")

manifest_path = os.path.join(tmp_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
PY

SHARD_FILES=()
while IFS= read -r shard_path; do
  SHARD_FILES+=("$shard_path")
done < <(python3 - "$TMP_DIR/manifest.json" <<'PY'
import json
import sys
for path in json.load(open(sys.argv[1])):
    print(path)
PY
)

echo "Launching ${#SHARD_FILES[@]} shard processes with JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}"

pids=()
logger_pids=()
shard_outputs=()
for shard_file in "${SHARD_FILES[@]}"; do
  shard_base="$(basename "${shard_file%.json}")"
  shard_output="$TMP_DIR/${shard_base}_out.json"
  shard_log="$TMP_DIR/${shard_base}.log"
  shard_pipe="$TMP_DIR/${shard_base}.pipe"
  shard_outputs+=("$shard_output")
  echo "  starting $shard_base"
  mkfifo "$shard_pipe"
  (
    tee "$shard_log" < "$shard_pipe" | awk -v prefix="[$shard_base] " '{ print prefix $0; fflush() }'
  ) &
  logger_pids+=("$!")
  shard_cmd=(
    julia --project=. scripts/utilities/reconstruct_model_costs.jl
    --exp "$EXP"
    --model "$MODEL_LABEL"
    --steps-file "$shard_file"
    "${REPLAY_TRACE_ARGS[@]}"
    --inference-file "$INFERENCE_FILE"
    --restrict-to-human-levels
    --human-costs-file "$HUMAN_COSTS_FILE"
    --problem-dir "$PROBLEM_DIR"
    --move-cost "$MOVE_COST"
    --interact-cost "$INTERACT_COST"
    --observe-cost "$OBSERVE_COST"
    --posterior-candidate-rule "$POSTERIOR_CANDIDATE_RULE"
    --posterior-mass-threshold "$POSTERIOR_MASS_THRESHOLD"
    --posterior-prob-threshold "$POSTERIOR_PROB_THRESHOLD"
    --output-file "$shard_output"
  )
  if [[ ${#EXP4_DISABLE_ARGS[@]} -gt 0 ]]; then
    shard_cmd+=("${EXP4_DISABLE_ARGS[@]}")
  fi
  "${shard_cmd[@]}" >"$shard_pipe" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    status=$?
    echo "Shard process failed: pid=$pid" >&2
    for other_pid in "${pids[@]}"; do
      if kill -0 "$other_pid" 2>/dev/null; then
        kill "$other_pid" 2>/dev/null || true
      fi
    done
    break
  fi
done

for logger_pid in "${logger_pids[@]}"; do
  wait "$logger_pid" || true
done

if [[ "$status" -ne 0 ]]; then
  echo "Shard logs:" >&2
  for shard_file in "${SHARD_FILES[@]}"; do
    shard_base="$(basename "${shard_file%.json}")"
    shard_log="$TMP_DIR/${shard_base}.log"
    echo "--- $shard_log ---" >&2
    tail -n 50 "$shard_log" >&2 || true
  done
  exit "$status"
fi

python3 - "$OUTPUT_FILE" "$EXP" "$MODEL_LABEL" "$STEPS_FILE" "$INFERENCE_FILE" "$PROBLEM_DIR" "$MOVE_COST" "$INTERACT_COST" "$OBSERVE_COST" "$POSTERIOR_CANDIDATE_RULE" "$POSTERIOR_MASS_THRESHOLD" "$POSTERIOR_PROB_THRESHOLD" "$JOBS" "$DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING" "${shard_outputs[@]}" <<'PY'
import json
import sys

output_file = sys.argv[1]
exp = sys.argv[2]
model_label = sys.argv[3]
steps_file = sys.argv[4]
inference_file = sys.argv[5]
problem_dir = sys.argv[6]
move_cost = float(sys.argv[7])
interact_cost = float(sys.argv[8])
observe_cost = float(sys.argv[9])
posterior_candidate_rule = sys.argv[10]
posterior_mass_threshold = float(sys.argv[11])
posterior_prob_threshold = float(sys.argv[12])
jobs = int(sys.argv[13])
disable_exp4_interaction_outcome_pruning = bool(int(sys.argv[14]))
shard_outputs = sys.argv[15:]

loaded = [json.load(open(path)) for path in shard_outputs]
first = loaded[0]
per_case = {}
plan_hits = 0
plan_misses = 0
plan_entries = 0
posterior_hits = 0
posterior_misses = 0
posterior_entries = 0
human_filter_enabled = False
human_costs_file = None
original_case_count = 0
kept_case_count = 0
matched_human_keys = set()

for shard in loaded:
    per_case.update(shard["per_case"])
    plan_stats = shard.get("replay_plan_cache_stats", {})
    posterior_stats = shard.get("posterior_filter_cache_stats", {})
    human_filter = shard.get("human_level_filter", {})
    plan_hits += int(plan_stats.get("hits", 0))
    plan_misses += int(plan_stats.get("misses", 0))
    plan_entries += int(plan_stats.get("entries", 0))
    posterior_hits += int(posterior_stats.get("hits", 0))
    posterior_misses += int(posterior_stats.get("misses", 0))
    posterior_entries += int(posterior_stats.get("entries", 0))
    human_filter_enabled = human_filter_enabled or bool(human_filter.get("enabled", False))
    if human_filter.get("human_costs_file"):
        human_costs_file = human_filter["human_costs_file"]
    original_case_count += int(human_filter.get("original_case_count", len(shard["per_case"])))
    kept_case_count += int(human_filter.get("kept_case_count", len(shard["per_case"])))
    matched_human_keys.update(human_filter.get("matched_human_keys", []))

values = list(per_case.values())

def mean(key):
    if not values:
        return 0.0
    return sum(float(v[key]) for v in values) / len(values)

summary = {
    "n_cases": len(values),
    "mean_total_cost": mean("total_cost"),
    "mean_observe_cost": mean("observe_cost"),
    "mean_planning_cost": mean("planning_cost"),
    "mean_total_steps": mean("total_steps"),
    "mean_observe_steps": mean("observe_steps"),
    "mean_planning_steps": mean("planning_steps"),
    "cases_with_warnings": sum(1 for v in values if v.get("warnings")),
}

plan_total = plan_hits + plan_misses
posterior_total = posterior_hits + posterior_misses

out = {
    "exp": first["exp"],
    "exp_normalized": first["exp_normalized"],
    "model_label": model_label,
    "steps_file": steps_file,
    "inference_file": inference_file,
    "problem_dir": problem_dir,
    "disable_exp4_interaction_outcome_pruning": disable_exp4_interaction_outcome_pruning,
    "posterior_candidate_rule": {
        "rule": posterior_candidate_rule,
        "mass_threshold": posterior_mass_threshold,
        "prob_threshold": posterior_prob_threshold,
    },
    "human_level_filter": {
        "enabled": human_filter_enabled,
        "human_costs_file": human_costs_file,
        "original_case_count": original_case_count,
        "kept_case_count": kept_case_count,
        "matched_human_keys": sorted(matched_human_keys),
    },
    "reconstruction_mode": first["reconstruction_mode"],
    "action_cost": {
        "move": move_cost,
        "interact": interact_cost,
        "observe": observe_cost,
    },
    "sharded_run": {
        "jobs_requested": jobs,
        "shards_completed": len(shard_outputs),
    },
    "replay_plan_cache_stats": {
        "hits": plan_hits,
        "misses": plan_misses,
        "hit_rate": (plan_hits / plan_total) if plan_total else 0.0,
        "entries": plan_entries,
    },
    "posterior_filter_cache_stats": {
        "hits": posterior_hits,
        "misses": posterior_misses,
        "hit_rate": (posterior_hits / posterior_total) if posterior_total else 0.0,
        "entries": posterior_entries,
    },
    "summary": summary,
    "per_case": per_case,
}

with open(output_file, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved merged reconstructed costs to: {output_file}")
print(f"Summary: {summary}")
PY
