#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ORGANIZED_OUTPUT_ROOT="${ORGANIZED_OUTPUT_ROOT:-scripts/organized_outputs}"
RUN_OUTPUT_DIR="$ORGANIZED_OUTPUT_ROOT/$RUN_ID/baselines"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_all_baselines.sh [--no-bootstrap]

Runs the 12 active baseline generators:
  - exp1: mentalize, non_mentalizing, naive
  - exp2: mentalize, non_mentalizing, naive
  - exp3: mentalize, non_mentalizing, naive
  - exp4: mentalize, non_mentalizing, naive
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

run_baseline_in_dir() {
  local label="$1"
  local run_dir="$2"
  local script_name="$3"

  echo "==> ${label}"
  echo "    dir:    ${run_dir}"
  echo "    script: ${script_name}"
  (
    cd "$ROOT_DIR/$run_dir"
    julia --project="$ROOT_DIR" "$script_name"
  )
}

run_baseline_in_dir "exp1 / social_mentalizing" "scripts/baselines/exp1" "baseline_mentalize_exp1.jl"
copy_output "model_outputs/baselines/exp1/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/social_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp1/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/social_mentalizing/replay_trace.json"
run_baseline_in_dir "exp1 / rational_non_mentalizing" "scripts/baselines/exp1" "baseline_non_mentalizing_exp1.jl"
copy_output "model_outputs/baselines/exp1/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/rational_non_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp1/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/rational_non_mentalizing/replay_trace.json"
run_baseline_in_dir "exp1 / naive_observer" "scripts/baselines/exp1" "baseline_naive_exp1.jl"
copy_output "model_outputs/baselines/exp1/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp1/naive_observer/steps_dict.json"
copy_output "model_outputs/baselines/exp1/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp1/naive_observer/replay_trace.json"

run_baseline_in_dir "exp2 / social_mentalizing" "scripts/baselines/exp2" "baseline_mentalize_exp2.jl"
copy_output "model_outputs/baselines/exp2/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/social_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp2/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/social_mentalizing/replay_trace.json"
run_baseline_in_dir "exp2 / rational_non_mentalizing" "scripts/baselines/exp2" "baseline_non_mentalizing_exp2.jl"
copy_output "model_outputs/baselines/exp2/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/rational_non_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp2/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/rational_non_mentalizing/replay_trace.json"
run_baseline_in_dir "exp2 / naive_observer" "scripts/baselines/exp2" "baseline_naive_exp2.jl"
copy_output "model_outputs/baselines/exp2/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp2/naive_observer/steps_dict.json"
copy_output "model_outputs/baselines/exp2/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp2/naive_observer/replay_trace.json"

run_baseline_in_dir "exp3 / social_mentalizing" "scripts/baselines/exp3" "baseline_mentalize_exp3.jl"
copy_output "model_outputs/baselines/exp3/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/social_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp3/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/social_mentalizing/replay_trace.json"
run_baseline_in_dir "exp3 / rational_non_mentalizing" "scripts/baselines/exp3" "baseline_non_mentalizing_exp3.jl"
copy_output "model_outputs/baselines/exp3/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/rational_non_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp3/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/rational_non_mentalizing/replay_trace.json"
run_baseline_in_dir "exp3 / naive_observer" "scripts/baselines/exp3" "baseline_naive_exp3.jl"
copy_output "model_outputs/baselines/exp3/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp3/naive_observer/steps_dict.json"
copy_output "model_outputs/baselines/exp3/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp3/naive_observer/replay_trace.json"

run_baseline_in_dir "exp4 / social_mentalizing" "scripts/baselines/exp4" "baseline_mentalize_exp4.jl"
copy_output "model_outputs/baselines/exp4/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/social_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp4/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/social_mentalizing/replay_trace.json"
run_baseline_in_dir "exp4 / rational_non_mentalizing" "scripts/baselines/exp4" "baseline_non_mentalizing_exp4.jl"
copy_output "model_outputs/baselines/exp4/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/rational_non_mentalizing/steps_dict.json"
copy_output "model_outputs/baselines/exp4/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/rational_non_mentalizing/replay_trace.json"
run_baseline_in_dir "exp4 / naive_observer" "scripts/baselines/exp4" "baseline_naive_exp4.jl"
copy_output "model_outputs/baselines/exp4/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp4/naive_observer/steps_dict.json"
copy_output "model_outputs/baselines/exp4/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp4/naive_observer/replay_trace.json"

echo "Organized outputs: $RUN_OUTPUT_DIR"
echo "Done."
