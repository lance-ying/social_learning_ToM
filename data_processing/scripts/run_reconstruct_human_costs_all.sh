#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-data_processing/outputs/human_costs}"
MOVE_COST="${MOVE_COST:-3}"
INTERACT_COST="${INTERACT_COST:-5}"
OBSERVE_COST="${OBSERVE_COST:-1}"

mkdir -p "$OUTPUT_DIR"

run_exp() {
  local exp="$1"
  echo "==> ${exp}"
  python3 data_processing/scripts/reconstruct_human_costs.py \
    --exp "$exp" \
    --move-cost "$MOVE_COST" \
    --interact-cost "$INTERACT_COST" \
    --observe-cost "$OBSERVE_COST" \
    --output-file "$OUTPUT_DIR/${exp}_human_costs.json"
}

run_exp exp1
run_exp exp2
run_exp exp3
run_exp exp4

echo "Done."
