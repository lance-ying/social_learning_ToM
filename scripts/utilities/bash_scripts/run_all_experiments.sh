#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ORGANIZED_OUTPUT_ROOT="${ORGANIZED_OUTPUT_ROOT:-scripts/organized_outputs}"
RUN_OUTPUT_DIR="$ORGANIZED_OUTPUT_ROOT/$RUN_ID/experiments"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_all_experiments.sh [--no-bootstrap]

Runs the active experiment generators:
  - exp1
  - exp2
  - exp3
  - exp4 wrapper
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bootstrap)
      BOOTSTRAP_ENV=0
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

if [[ "$BOOTSTRAP_ENV" == "1" ]]; then
  echo "==> bootstrapping Julia environment"
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
fi

mkdir -p "$RUN_OUTPUT_DIR"

copy_output() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "Missing expected output: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

run_experiment() {
  local label="$1"
  local script_path="$2"

  echo "==> ${label}"
  echo "    script: ${script_path}"
  julia --project=. "$script_path"
}

run_experiment "exp1 / full experiment" "scripts/experiments/run_experiment_exp1.jl"
copy_output "model_outputs/experiments/exp1/steps_dict.json" "$RUN_OUTPUT_DIR/exp1/steps_dict.json"
copy_output "model_outputs/experiments/exp1/replay_trace.json" "$RUN_OUTPUT_DIR/exp1/replay_trace.json"

run_experiment "exp2 / full experiment" "scripts/experiments/run_experiment_exp2.jl"
copy_output "model_outputs/experiments/exp2/steps_dict.json" "$RUN_OUTPUT_DIR/exp2/steps_dict.json"
copy_output "model_outputs/experiments/exp2/replay_trace.json" "$RUN_OUTPUT_DIR/exp2/replay_trace.json"

run_experiment "exp3 / full experiment" "scripts/experiments/run_experiment_exp3.jl"
copy_output "model_outputs/experiments/exp3/steps_dict.json" "$RUN_OUTPUT_DIR/exp3/steps_dict.json"
copy_output "model_outputs/experiments/exp3/replay_trace.json" "$RUN_OUTPUT_DIR/exp3/replay_trace.json"

run_experiment "exp4 / wrapper" "scripts/experiments/run_experiment_exp4_wrapper.jl"
copy_output "model_outputs/experiments/exp4/steps_dict.json" "$RUN_OUTPUT_DIR/exp4/steps_dict.json"
copy_output "model_outputs/experiments/exp4/scenario1/steps_dict.json" "$RUN_OUTPUT_DIR/exp4/scenario1_steps_dict.json"
copy_output "model_outputs/experiments/exp4/scenario2/steps_dict.json" "$RUN_OUTPUT_DIR/exp4/scenario2_steps_dict.json"
copy_output "model_outputs/experiments/exp4/replay_trace.json" "$RUN_OUTPUT_DIR/exp4/replay_trace.json"
copy_output "model_outputs/experiments/exp4/scenario1/replay_trace.json" "$RUN_OUTPUT_DIR/exp4/scenario1_replay_trace.json"
copy_output "model_outputs/experiments/exp4/scenario2/replay_trace.json" "$RUN_OUTPUT_DIR/exp4/scenario2_replay_trace.json"

echo "Organized outputs: $RUN_OUTPUT_DIR"
echo "Done."
