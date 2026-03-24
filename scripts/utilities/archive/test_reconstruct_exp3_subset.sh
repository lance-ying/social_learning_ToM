#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

N_CASES="${1:-1}"
MODEL_LABEL="${2:-full_model}"
STEPS_SRC="${STEPS_SRC:-steps_dict_exp3.json}"
INFERENCE_FILE="${INFERENCE_FILE:-data/inference/inference_data_exp3.jld2}"
PROBLEM_DIR="${PROBLEM_DIR:-dataset/problems_exp3}"
MOVE_COST="${MOVE_COST:-3}"
INTERACT_COST="${INTERACT_COST:-5}"
OBSERVE_COST="${OBSERVE_COST:-1}"
STEPS_OUT="${STEPS_OUT:-/tmp/exp3_subset_${N_CASES}_${MODEL_LABEL}.json}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/exp3_subset_${N_CASES}_${MODEL_LABEL}_replay.json}"

if ! [[ "$N_CASES" =~ ^[0-9]+$ ]] || [[ "$N_CASES" -lt 1 ]]; then
  echo "N_CASES must be a positive integer, got: $N_CASES" >&2
  exit 1
fi

python3 -c 'import json, sys; from collections import OrderedDict; src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); data = json.load(open(src), object_pairs_hook=OrderedDict); subset = OrderedDict(list(data.items())[:n]); json.dump(subset, open(out, "w"), indent=2); print("subset keys:", list(subset.keys()))' \
  "$STEPS_SRC" "$STEPS_OUT" "$N_CASES"

echo "Running exp3 replay subset"
echo "  cases:       $N_CASES"
echo "  model:       $MODEL_LABEL"
echo "  steps out:   $STEPS_OUT"
echo "  output file: $OUTPUT_FILE"

julia --project=. scripts/utilities/reconstruct_model_costs.jl \
  --exp exp3 \
  --model "$MODEL_LABEL" \
  --steps-file "$STEPS_OUT" \
  --inference-file "$INFERENCE_FILE" \
  --problem-dir "$PROBLEM_DIR" \
  --move-cost "$MOVE_COST" \
  --interact-cost "$INTERACT_COST" \
  --observe-cost "$OBSERVE_COST" \
  --output-file "$OUTPUT_FILE"
