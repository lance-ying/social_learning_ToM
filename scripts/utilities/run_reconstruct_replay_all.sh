#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-scripts/experiments/experiment_outputs/reconstructed_costs_replay}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
MOVE_COST="${MOVE_COST:-3}"
INTERACT_COST="${INTERACT_COST:-5}"
OBSERVE_COST="${OBSERVE_COST:-1}"
EXPERIMENTS="exp1,exp2,exp3,exp4"
PARALLEL_MULTIAGENT_JOBS="${PARALLEL_MULTIAGENT_JOBS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_reconstruct_replay_all.sh [--experiments exp1,exp2,...] [--exp3-4] [--parallel-multiagent-jobs N] [--no-bootstrap]

Examples:
  bash scripts/utilities/run_reconstruct_replay_all.sh
  bash scripts/utilities/run_reconstruct_replay_all.sh --exp3-4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp3,exp4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp2
  bash scripts/utilities/run_reconstruct_replay_all.sh --exp3-4 --parallel-multiagent-jobs 4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp1,exp2 --parallel-multiagent-jobs 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiments)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --experiments" >&2
        exit 1
      fi
      EXPERIMENTS="$2"
      shift 2
      ;;
    --exp3-4)
      EXPERIMENTS="exp3,exp4"
      shift
      ;;
    --parallel-multiagent-jobs)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --parallel-multiagent-jobs" >&2
        exit 1
      fi
      PARALLEL_MULTIAGENT_JOBS="$2"
      shift 2
      ;;
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

# Exp1
EXP1_MODEL_STEPS="${EXP1_MODEL_STEPS:-steps.dict.json}"
EXP1_MENTALIZE_STEPS="${EXP1_MENTALIZE_STEPS:-step_dict_mentalize_exp1.json}"
EXP1_NONMENTALIZE_STEPS="${EXP1_NONMENTALIZE_STEPS:-step_dict_nonmentalize_exp1.json}"
EXP1_NAIVE_STEPS="${EXP1_NAIVE_STEPS:-scripts/baselines/exp1/step_dict_naive_exp1.json}"
EXP1_INFERENCE="${EXP1_INFERENCE:-data/inference/inference_data_exp1.jld2}"
EXP1_PROBLEM_DIR="${EXP1_PROBLEM_DIR:-dataset/problems_exp1}"
EXP1_HUMAN_COSTS="${EXP1_HUMAN_COSTS:-data_processing/outputs/human_costs/exp1_human_costs.json}"

# Exp2
EXP2_MODEL_STEPS="${EXP2_MODEL_STEPS:-steps_exp2.json}"
EXP2_MENTALIZE_STEPS="${EXP2_MENTALIZE_STEPS:-step_dict_mentalize_exp2.json}"
EXP2_NONMENTALIZE_STEPS="${EXP2_NONMENTALIZE_STEPS:-scripts/baselines/exp2/step_dict_nonmentalize_exp2.json}"
EXP2_NAIVE_STEPS="${EXP2_NAIVE_STEPS:-scripts/baselines/exp2/step_dict_naive_exp2.json}"
EXP2_INFERENCE="${EXP2_INFERENCE:-data/inference/inference_data_exp2.jld2}"
EXP2_PROBLEM_DIR="${EXP2_PROBLEM_DIR:-dataset/problems_exp2}"
EXP2_HUMAN_COSTS="${EXP2_HUMAN_COSTS:-data_processing/outputs/human_costs/exp2_human_costs.json}"

# Exp3
EXP3_MODEL_STEPS="${EXP3_MODEL_STEPS:-scripts/experiments/experiment_outputs/step_dict_3_031625.json}"
EXP3_MENTALIZE_STEPS="${EXP3_MENTALIZE_STEPS:-step_dict_mentalize_exp3.json}"
EXP3_NONMENTALIZE_STEPS="${EXP3_NONMENTALIZE_STEPS:-scripts/baselines/exp3/step_dict_nonmentalize_exp3_v2.json}"
EXP3_NAIVE_STEPS="${EXP3_NAIVE_STEPS:-step_dict_naive_exp3.json}"
EXP3_INFERENCE="${EXP3_INFERENCE:-data/inference/inference_data_exp3.jld2}"
EXP3_PROBLEM_DIR="${EXP3_PROBLEM_DIR:-dataset/problems_exp3}"
EXP3_HUMAN_COSTS="${EXP3_HUMAN_COSTS:-data_processing/outputs/human_costs/exp3_human_costs.json}"

# Exp4
EXP4_MODEL_STEPS="${EXP4_MODEL_STEPS:-scripts/experiments/experiment_outputs/steps_dict_exp4_031726_2.json}"
EXP4_MENTALIZE_STEPS="${EXP4_MENTALIZE_STEPS:-step_dict_mentalize_exp4.json}"
EXP4_NONMENTALIZE_STEPS="${EXP4_NONMENTALIZE_STEPS:-step_dict_nonmentalize_exp4.json}"
EXP4_NAIVE_STEPS="${EXP4_NAIVE_STEPS:-step_dict_naive_exp4.json}"
EXP4_INFERENCE="${EXP4_INFERENCE:-data/inference/inference_exp4_020126_1.jld2}"
EXP4_PROBLEM_DIR="${EXP4_PROBLEM_DIR:-dataset/problems_exp4_013026}"
EXP4_HUMAN_COSTS="${EXP4_HUMAN_COSTS:-data_processing/outputs/human_costs/exp4_human_costs.json}"

mkdir -p "$OUTPUT_DIR"

if [[ "$BOOTSTRAP_ENV" == "1" ]]; then
  echo "==> bootstrapping Julia environment"
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
fi

run_reconstruct() {
  local exp="$1"
  local label="$2"
  local steps_file="$3"
  local inference_file="$4"
  local problem_dir="$5"
  local human_costs_file="$6"
  local output_file="$OUTPUT_DIR/${exp}_${label}.json"

  echo "==> ${exp} / ${label}"
  echo "    steps:   $steps_file"
  echo "    output:  $output_file"

  if [[ "$PARALLEL_MULTIAGENT_JOBS" -gt 1 ]]; then
    bash scripts/utilities/run_reconstruct_sharded.sh \
      --exp "$exp" \
      --model "$label" \
      --steps-file "$steps_file" \
      --inference-file "$inference_file" \
      --human-costs-file "$human_costs_file" \
      --problem-dir "$problem_dir" \
      --move-cost "$MOVE_COST" \
      --interact-cost "$INTERACT_COST" \
      --observe-cost "$OBSERVE_COST" \
      --jobs "$PARALLEL_MULTIAGENT_JOBS" \
      --output-file "$output_file"
  else
    julia --project=. scripts/utilities/reconstruct_model_costs.jl \
      --exp "$exp" \
      --model "$label" \
      --steps-file "$steps_file" \
      --inference-file "$inference_file" \
      --restrict-to-human-levels \
      --human-costs-file "$human_costs_file" \
      --problem-dir "$problem_dir" \
      --move-cost "$MOVE_COST" \
      --interact-cost "$INTERACT_COST" \
      --observe-cost "$OBSERVE_COST" \
      --output-file "$output_file"
  fi
}

has_experiment() {
  local exp="$1"
  local item
  local normalized=",${EXPERIMENTS// /},"
  IFS=',' read -r -a items <<< "$EXPERIMENTS"
  for item in "${items[@]}"; do
    if [[ "$item" == "$exp" ]]; then
      return 0
    fi
  done
  return 1
}

echo "Output directory: $OUTPUT_DIR"
echo "Action costs: move=$MOVE_COST interact=$INTERACT_COST observe=$OBSERVE_COST"
echo "Experiments: $EXPERIMENTS"
echo "Parallel multi-agent jobs: $PARALLEL_MULTIAGENT_JOBS"

if has_experiment exp1; then
  run_reconstruct exp1 full_model "$EXP1_MODEL_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 social_mentalizing "$EXP1_MENTALIZE_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 rational_non_mentalizing "$EXP1_NONMENTALIZE_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 naive_observer "$EXP1_NAIVE_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
fi

if has_experiment exp2; then
  run_reconstruct exp2 full_model "$EXP2_MODEL_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 social_mentalizing "$EXP2_MENTALIZE_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 rational_non_mentalizing "$EXP2_NONMENTALIZE_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 naive_observer "$EXP2_NAIVE_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
fi

if has_experiment exp3; then
  run_reconstruct exp3 full_model "$EXP3_MODEL_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 social_mentalizing "$EXP3_MENTALIZE_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 rational_non_mentalizing "$EXP3_NONMENTALIZE_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 naive_observer "$EXP3_NAIVE_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
fi

if has_experiment exp4; then
  run_reconstruct exp4 full_model "$EXP4_MODEL_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 social_mentalizing "$EXP4_MENTALIZE_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 rational_non_mentalizing "$EXP4_NONMENTALIZE_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 naive_observer "$EXP4_NAIVE_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
fi

echo "Done."
