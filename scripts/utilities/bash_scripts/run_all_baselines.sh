#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

JULIA_BIN="${JULIA_BIN:-julia +1.11.9}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ORGANIZED_OUTPUT_ROOT="${ORGANIZED_OUTPUT_ROOT:-scripts/organized_outputs}"
RUN_OUTPUT_DIR="$ORGANIZED_OUTPUT_ROOT/$RUN_ID/baselines"
SELECTED_EXPERIMENTS=("exp1" "exp2" "exp3" "exp4")
SELECTED_MODELS=("social_mentalizing" "rational_non_mentalizing" "naive_observer")
MODEL_FILTER_SET=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_all_baselines.sh [--no-bootstrap] [--exp <exp1|exp2|exp3|exp4>] [--exp34] [--model <social_mentalizing|rational_non_mentalizing|naive_observer|agent1_naive_planner>[,...]]

Runs the 12 active baseline generators:
  - exp1: mentalize, non_mentalizing, naive
  - exp2: mentalize, non_mentalizing, naive
  - exp3: mentalize, non_mentalizing, naive
  - exp4: mentalize, non_mentalizing, naive

Notes:
  - agent1_naive_planner has no standalone baseline generator.
  - Selecting --model agent1_naive_planner runs only the naive_observer baseline,
    which provides the replay-trace inputs used by agent1_naive_planner reconstruction.
  - --model may be passed multiple times or as a comma-separated list.
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

has_selected_model() {
  local target="$1"
  local model
  for model in "${SELECTED_MODELS[@]-}"; do
    if [[ "$model" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

should_run_baseline_model() {
  local target="$1"
  if has_selected_model "$target"; then
    return 0
  fi
  if [[ "$target" == "naive_observer" ]] && has_selected_model "agent1_naive_planner"; then
    return 0
  fi
  return 1
}

is_valid_model() {
  case "$1" in
    social_mentalizing|rational_non_mentalizing|naive_observer|agent1_naive_planner)
      return 0
      ;;
  esac
  return 1
}

append_selected_model() {
  local candidate="$1"
  if ! is_valid_model "$candidate"; then
    echo "Unknown model: $candidate" >&2
    usage >&2
    exit 1
  fi
  if ! has_selected_model "$candidate"; then
    SELECTED_MODELS+=("$candidate")
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
      case "$2" in
        exp1|exp2|exp3|exp4)
          SELECTED_EXPERIMENTS=("$2")
          ;;
        *)
          echo "Unknown experiment: $2" >&2
          usage >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --exp34)
      SELECTED_EXPERIMENTS=("exp3" "exp4")
      shift
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --model" >&2
        usage >&2
        exit 1
      fi
      if [[ "$MODEL_FILTER_SET" == "0" ]]; then
        SELECTED_MODELS=()
        MODEL_FILTER_SET=1
      fi
      IFS=',' read -r -a requested_models <<< "$2"
      for requested_model in "${requested_models[@]}"; do
        append_selected_model "$requested_model"
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

if has_selected_model "agent1_naive_planner"; then
  echo "Note: agent1_naive_planner has no standalone baseline script; running naive_observer only to refresh its reconstruction inputs."
fi

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
    bash -lc "$JULIA_BIN --project=\"$ROOT_DIR\" \"$script_name\""
  )
}

if has_experiment "exp1"; then
  if should_run_baseline_model "social_mentalizing"; then
    run_baseline_in_dir "exp1 / social_mentalizing" "scripts/baselines/exp1" "baseline_mentalize_exp1.jl"
    copy_output "model_outputs/baselines/exp1/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/social_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp1/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/social_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "rational_non_mentalizing"; then
    run_baseline_in_dir "exp1 / rational_non_mentalizing" "scripts/baselines/exp1" "baseline_non_mentalizing_exp1.jl"
    copy_output "model_outputs/baselines/exp1/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/rational_non_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp1/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp1/rational_non_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "naive_observer"; then
    run_baseline_in_dir "exp1 / naive_observer" "scripts/baselines/exp1" "baseline_naive_exp1.jl"
    copy_output "model_outputs/baselines/exp1/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp1/naive_observer/steps_dict.json"
    copy_output "model_outputs/baselines/exp1/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp1/naive_observer/replay_trace.json"
  fi
fi

if has_experiment "exp2"; then
  if should_run_baseline_model "social_mentalizing"; then
    run_baseline_in_dir "exp2 / social_mentalizing" "scripts/baselines/exp2" "baseline_mentalize_exp2.jl"
    copy_output "model_outputs/baselines/exp2/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/social_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp2/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/social_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "rational_non_mentalizing"; then
    run_baseline_in_dir "exp2 / rational_non_mentalizing" "scripts/baselines/exp2" "baseline_non_mentalizing_exp2.jl"
    copy_output "model_outputs/baselines/exp2/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/rational_non_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp2/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp2/rational_non_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "naive_observer"; then
    run_baseline_in_dir "exp2 / naive_observer" "scripts/baselines/exp2" "baseline_naive_exp2.jl"
    copy_output "model_outputs/baselines/exp2/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp2/naive_observer/steps_dict.json"
    copy_output "model_outputs/baselines/exp2/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp2/naive_observer/replay_trace.json"
  fi
fi

if has_experiment "exp3"; then
  if should_run_baseline_model "social_mentalizing"; then
    run_baseline_in_dir "exp3 / social_mentalizing" "scripts/baselines/exp3" "baseline_mentalize_exp3.jl"
    copy_output "model_outputs/baselines/exp3/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/social_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp3/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/social_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "rational_non_mentalizing"; then
    run_baseline_in_dir "exp3 / rational_non_mentalizing" "scripts/baselines/exp3" "baseline_non_mentalizing_exp3.jl"
    copy_output "model_outputs/baselines/exp3/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/rational_non_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp3/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp3/rational_non_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "naive_observer"; then
    run_baseline_in_dir "exp3 / naive_observer" "scripts/baselines/exp3" "baseline_naive_exp3.jl"
    copy_output "model_outputs/baselines/exp3/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp3/naive_observer/steps_dict.json"
    copy_output "model_outputs/baselines/exp3/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp3/naive_observer/replay_trace.json"
  fi
fi

if has_experiment "exp4"; then
  if should_run_baseline_model "social_mentalizing"; then
    run_baseline_in_dir "exp4 / social_mentalizing" "scripts/baselines/exp4" "baseline_mentalize_exp4.jl"
    copy_output "model_outputs/baselines/exp4/step_dict_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/social_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp4/replay_trace_social_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/social_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "rational_non_mentalizing"; then
    run_baseline_in_dir "exp4 / rational_non_mentalizing" "scripts/baselines/exp4" "baseline_non_mentalizing_exp4.jl"
    copy_output "model_outputs/baselines/exp4/step_dict_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/rational_non_mentalizing/steps_dict.json"
    copy_output "model_outputs/baselines/exp4/replay_trace_rational_non_mentalizing.json" "$RUN_OUTPUT_DIR/exp4/rational_non_mentalizing/replay_trace.json"
  fi
  if should_run_baseline_model "naive_observer"; then
    run_baseline_in_dir "exp4 / naive_observer" "scripts/baselines/exp4" "baseline_naive_exp4.jl"
    copy_output "model_outputs/baselines/exp4/step_dict_naive_observer.json" "$RUN_OUTPUT_DIR/exp4/naive_observer/steps_dict.json"
    copy_output "model_outputs/baselines/exp4/replay_trace_naive_observer.json" "$RUN_OUTPUT_DIR/exp4/naive_observer/replay_trace.json"
  fi
fi

echo "Organized outputs: $RUN_OUTPUT_DIR"
echo "Done."
