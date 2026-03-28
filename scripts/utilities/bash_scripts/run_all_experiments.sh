#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

JULIA_BIN="${JULIA_BIN:-julia +1.11.9}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ORGANIZED_OUTPUT_ROOT="${ORGANIZED_OUTPUT_ROOT:-scripts/organized_outputs}"
RUN_OUTPUT_DIR="$ORGANIZED_OUTPUT_ROOT/$RUN_ID/experiments"
SELECTED_EXPERIMENTS=("exp1" "exp2" "exp3" "exp4")

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_all_experiments.sh [--no-bootstrap] [--exp <exp1|exp2|exp3|exp4>[,...]]

Runs the active experiment generators:
  - exp1
  - exp2
  - exp3
  - exp4 wrapper

Notes:
  - `--exp` may be passed multiple times or as a comma-separated list.
  - Examples:
      bash scripts/utilities/bash_scripts/run_all_experiments.sh --exp exp2
      bash scripts/utilities/bash_scripts/run_all_experiments.sh --exp exp2,exp4
EOF
}

has_experiment() {
  local target="$1"
  local exp
  for exp in "${SELECTED_EXPERIMENTS[@]-}"; do
    if [[ "$exp" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

append_selected_experiment() {
  local candidate="$1"
  case "$candidate" in
    exp1|exp2|exp3|exp4)
      ;;
    *)
      echo "Unknown experiment: $candidate" >&2
      usage >&2
      exit 1
      ;;
  esac
  if ! has_experiment "$candidate"; then
    SELECTED_EXPERIMENTS+=("$candidate")
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bootstrap)
      BOOTSTRAP_ENV=0
      shift
      ;;
    --exp)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --exp" >&2
        usage >&2
        exit 1
      fi
      IFS=',' read -r -a requested_experiments <<< "$2"
      if [[ ${#requested_experiments[@]} -eq 0 ]]; then
        echo "Missing value for --exp" >&2
        usage >&2
        exit 1
      fi
      if [[ ${#SELECTED_EXPERIMENTS[@]} -eq 4 ]]; then
        SELECTED_EXPERIMENTS=()
      fi
      for requested_experiment in "${requested_experiments[@]}"; do
        append_selected_experiment "$requested_experiment"
      done
      shift 2
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
  bash -lc "$JULIA_BIN --project=. -e 'using Pkg; Pkg.instantiate()'"
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
  bash -lc "$JULIA_BIN --project=. \"$script_path\""
}

if has_experiment "exp1"; then
  run_experiment "exp1 / full experiment" "scripts/experiments/run_experiment_exp1.jl"
  copy_output "model_outputs/experiments/exp1/steps_dict.json" "$RUN_OUTPUT_DIR/exp1/steps_dict.json"
  copy_output "model_outputs/experiments/exp1/replay_trace.json" "$RUN_OUTPUT_DIR/exp1/replay_trace.json"
fi

if has_experiment "exp2"; then
  run_experiment "exp2 / full experiment" "scripts/experiments/run_experiment_exp2.jl"
  copy_output "model_outputs/experiments/exp2/steps_dict.json" "$RUN_OUTPUT_DIR/exp2/steps_dict.json"
  copy_output "model_outputs/experiments/exp2/replay_trace.json" "$RUN_OUTPUT_DIR/exp2/replay_trace.json"
fi

if has_experiment "exp3"; then
  run_experiment "exp3 / full experiment" "scripts/experiments/run_experiment_exp3.jl"
  copy_output "model_outputs/experiments/exp3/steps_dict.json" "$RUN_OUTPUT_DIR/exp3/steps_dict.json"
  copy_output "model_outputs/experiments/exp3/replay_trace.json" "$RUN_OUTPUT_DIR/exp3/replay_trace.json"
fi

if has_experiment "exp4"; then
  run_experiment "exp4 / wrapper" "scripts/experiments/run_experiment_exp4_wrapper.jl"
  copy_output "model_outputs/experiments/exp4/steps_dict.json" "$RUN_OUTPUT_DIR/exp4/steps_dict.json"
  copy_output "model_outputs/experiments/exp4/scenario1/steps_dict.json" "$RUN_OUTPUT_DIR/exp4/scenario1_steps_dict.json"
  copy_output "model_outputs/experiments/exp4/scenario2/steps_dict.json" "$RUN_OUTPUT_DIR/exp4/scenario2_steps_dict.json"
  copy_output "model_outputs/experiments/exp4/replay_trace.json" "$RUN_OUTPUT_DIR/exp4/replay_trace.json"
  copy_output "model_outputs/experiments/exp4/scenario1/replay_trace.json" "$RUN_OUTPUT_DIR/exp4/scenario1_replay_trace.json"
  copy_output "model_outputs/experiments/exp4/scenario2/replay_trace.json" "$RUN_OUTPUT_DIR/exp4/scenario2_replay_trace.json"
fi

echo "Organized outputs: $RUN_OUTPUT_DIR"
echo "Done."
