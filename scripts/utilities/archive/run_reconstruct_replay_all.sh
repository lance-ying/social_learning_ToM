#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-scripts/experiments/experiment_outputs/reconstructed_costs_replay}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
MOVE_COST="${MOVE_COST:-3}"
INTERACT_COST="${INTERACT_COST:-5}"
OBSERVE_COST="${OBSERVE_COST:-1}"
POSTERIOR_CANDIDATE_RULE="${POSTERIOR_CANDIDATE_RULE:-prob_threshold}"
POSTERIOR_MASS_THRESHOLD="${POSTERIOR_MASS_THRESHOLD:-0.9}"
POSTERIOR_PROB_THRESHOLD="${POSTERIOR_PROB_THRESHOLD:-0.1}"
EXPERIMENTS="exp1,exp2,exp3,exp4"
PARALLEL_MULTIAGENT_JOBS="${PARALLEL_MULTIAGENT_JOBS:-1}"
DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING="${DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_reconstruct_replay_all.sh [--experiments exp1,exp2,...] [--exp3-4] [--exp4] [--parallel-multiagent-jobs N] [--disable-exp4-interaction-outcome-pruning] [--no-bootstrap]

Examples:
  bash scripts/utilities/run_reconstruct_replay_all.sh
  bash scripts/utilities/run_reconstruct_replay_all.sh --exp3-4
  bash scripts/utilities/run_reconstruct_replay_all.sh --exp4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp3,exp4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp2
  bash scripts/utilities/run_reconstruct_replay_all.sh --exp3-4 --parallel-multiagent-jobs 4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp1,exp2 --parallel-multiagent-jobs 4
  bash scripts/utilities/run_reconstruct_replay_all.sh --experiments exp4 --disable-exp4-interaction-outcome-pruning

Environment overrides:
  POSTERIOR_CANDIDATE_RULE=prob_threshold|top_mass|positive_support
  POSTERIOR_MASS_THRESHOLD=0.9
  POSTERIOR_PROB_THRESHOLD=0.1
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
    --exp4)
      EXPERIMENTS="exp4"
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
    --disable-exp4-interaction-outcome-pruning)
      DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING=1
      shift
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
EXP1_MODEL_STEPS="${EXP1_MODEL_STEPS:-scripts/experiments/outputs/exp1/steps_dict.json}"
EXP1_MODEL_REPLAY_TRACE="${EXP1_MODEL_REPLAY_TRACE:-scripts/experiments/outputs/exp1/replay_trace.json}"
EXP1_MENTALIZE_STEPS="${EXP1_MENTALIZE_STEPS:-scripts/baselines/outputs/exp1/step_dict_social_mentalizing.json}"
EXP1_MENTALIZE_REPLAY_TRACE="${EXP1_MENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp1/replay_trace_social_mentalizing.json}"
EXP1_NONMENTALIZE_STEPS="${EXP1_NONMENTALIZE_STEPS:-scripts/baselines/outputs/exp1/step_dict_rational_non_mentalizing.json}"
EXP1_NONMENTALIZE_REPLAY_TRACE="${EXP1_NONMENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp1/replay_trace_rational_non_mentalizing.json}"
EXP1_NAIVE_STEPS="${EXP1_NAIVE_STEPS:-scripts/baselines/outputs/exp1/step_dict_naive_observer.json}"
EXP1_NAIVE_REPLAY_TRACE="${EXP1_NAIVE_REPLAY_TRACE:-scripts/baselines/outputs/exp1/replay_trace_naive_observer.json}"
EXP1_AGENT1_NAIVE_STEPS="${EXP1_AGENT1_NAIVE_STEPS:-$EXP1_NAIVE_STEPS}"
EXP1_AGENT1_NAIVE_REPLAY_TRACE="${EXP1_AGENT1_NAIVE_REPLAY_TRACE:-$EXP1_NAIVE_REPLAY_TRACE}"
EXP1_INFERENCE="${EXP1_INFERENCE:-data/inference/inference_data_exp1.jld2}"
EXP1_PROBLEM_DIR="${EXP1_PROBLEM_DIR:-dataset/problems_exp1}"
EXP1_HUMAN_COSTS="${EXP1_HUMAN_COSTS:-data_processing/outputs/human_costs/exp1_human_costs.json}"

# Exp2
EXP2_MODEL_STEPS="${EXP2_MODEL_STEPS:-scripts/experiments/outputs/exp2/steps_dict.json}"
EXP2_MODEL_REPLAY_TRACE="${EXP2_MODEL_REPLAY_TRACE:-scripts/experiments/outputs/exp2/replay_trace.json}"
EXP2_MENTALIZE_STEPS="${EXP2_MENTALIZE_STEPS:-scripts/baselines/outputs/exp2/step_dict_social_mentalizing.json}"
EXP2_MENTALIZE_REPLAY_TRACE="${EXP2_MENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp2/replay_trace_social_mentalizing.json}"
EXP2_NONMENTALIZE_STEPS="${EXP2_NONMENTALIZE_STEPS:-scripts/baselines/outputs/exp2/step_dict_rational_non_mentalizing.json}"
EXP2_NONMENTALIZE_REPLAY_TRACE="${EXP2_NONMENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp2/replay_trace_rational_non_mentalizing.json}"
EXP2_NAIVE_STEPS="${EXP2_NAIVE_STEPS:-scripts/baselines/outputs/exp2/step_dict_naive_observer.json}"
EXP2_NAIVE_REPLAY_TRACE="${EXP2_NAIVE_REPLAY_TRACE:-scripts/baselines/outputs/exp2/replay_trace_naive_observer.json}"
EXP2_AGENT1_NAIVE_STEPS="${EXP2_AGENT1_NAIVE_STEPS:-$EXP2_NAIVE_STEPS}"
EXP2_AGENT1_NAIVE_REPLAY_TRACE="${EXP2_AGENT1_NAIVE_REPLAY_TRACE:-$EXP2_NAIVE_REPLAY_TRACE}"
EXP2_INFERENCE="${EXP2_INFERENCE:-data/inference/inference_data_exp2.jld2}"
EXP2_PROBLEM_DIR="${EXP2_PROBLEM_DIR:-dataset/problems_exp2}"
EXP2_HUMAN_COSTS="${EXP2_HUMAN_COSTS:-data_processing/outputs/human_costs/exp2_human_costs.json}"

# Exp3
EXP3_MODEL_STEPS="${EXP3_MODEL_STEPS:-scripts/experiments/outputs/exp3/steps_dict.json}"
EXP3_MODEL_REPLAY_TRACE="${EXP3_MODEL_REPLAY_TRACE:-scripts/experiments/outputs/exp3/replay_trace.json}"
EXP3_MENTALIZE_STEPS="${EXP3_MENTALIZE_STEPS:-scripts/baselines/outputs/exp3/step_dict_social_mentalizing.json}"
EXP3_MENTALIZE_REPLAY_TRACE="${EXP3_MENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp3/replay_trace_social_mentalizing.json}"
EXP3_NONMENTALIZE_STEPS="${EXP3_NONMENTALIZE_STEPS:-scripts/baselines/outputs/exp3/step_dict_rational_non_mentalizing.json}"
EXP3_NONMENTALIZE_REPLAY_TRACE="${EXP3_NONMENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp3/replay_trace_rational_non_mentalizing.json}"
EXP3_NAIVE_STEPS="${EXP3_NAIVE_STEPS:-scripts/baselines/outputs/exp3/step_dict_naive_observer.json}"
EXP3_NAIVE_REPLAY_TRACE="${EXP3_NAIVE_REPLAY_TRACE:-scripts/baselines/outputs/exp3/replay_trace_naive_observer.json}"
EXP3_AGENT1_NAIVE_STEPS="${EXP3_AGENT1_NAIVE_STEPS:-$EXP3_NAIVE_STEPS}"
EXP3_AGENT1_NAIVE_REPLAY_TRACE="${EXP3_AGENT1_NAIVE_REPLAY_TRACE:-$EXP3_NAIVE_REPLAY_TRACE}"
EXP3_INFERENCE="${EXP3_INFERENCE:-data/inference/inference_data_exp3.jld2}"
EXP3_PROBLEM_DIR="${EXP3_PROBLEM_DIR:-dataset/problems_exp3}"
EXP3_HUMAN_COSTS="${EXP3_HUMAN_COSTS:-data_processing/outputs/human_costs/exp3_human_costs.json}"

# Exp4
EXP4_MODEL_STEPS="${EXP4_MODEL_STEPS:-scripts/experiments/outputs/exp4/steps_dict.json}"
EXP4_MODEL_REPLAY_TRACE="${EXP4_MODEL_REPLAY_TRACE:-scripts/experiments/outputs/exp4/replay_trace.json}"
EXP4_MENTALIZE_STEPS="${EXP4_MENTALIZE_STEPS:-scripts/baselines/outputs/exp4/step_dict_social_mentalizing.json}"
EXP4_MENTALIZE_REPLAY_TRACE="${EXP4_MENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp4/replay_trace_social_mentalizing.json}"
EXP4_NONMENTALIZE_STEPS="${EXP4_NONMENTALIZE_STEPS:-scripts/baselines/outputs/exp4/step_dict_rational_non_mentalizing.json}"
EXP4_NONMENTALIZE_REPLAY_TRACE="${EXP4_NONMENTALIZE_REPLAY_TRACE:-scripts/baselines/outputs/exp4/replay_trace_rational_non_mentalizing.json}"
EXP4_NAIVE_STEPS="${EXP4_NAIVE_STEPS:-scripts/baselines/outputs/exp4/step_dict_naive_observer.json}"
EXP4_NAIVE_REPLAY_TRACE="${EXP4_NAIVE_REPLAY_TRACE:-scripts/baselines/outputs/exp4/replay_trace_naive_observer.json}"
EXP4_AGENT1_NAIVE_STEPS="${EXP4_AGENT1_NAIVE_STEPS:-$EXP4_NAIVE_STEPS}"
EXP4_AGENT1_NAIVE_REPLAY_TRACE="${EXP4_AGENT1_NAIVE_REPLAY_TRACE:-$EXP4_NAIVE_REPLAY_TRACE}"
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
  local replay_trace_file="$4"
  local inference_file="$5"
  local problem_dir="$6"
  local human_costs_file="$7"
  local output_file="$OUTPUT_DIR/${exp}_${label}.json"

  echo "==> ${exp} / ${label}"
  echo "    steps:   $steps_file"
  if [[ -n "$replay_trace_file" ]]; then
    echo "    replay:  $replay_trace_file"
  fi
  echo "    output:  $output_file"

  local replay_trace_args=()
  if [[ -n "$replay_trace_file" ]]; then
    replay_trace_args=(--replay-trace-file "$replay_trace_file")
  fi
  local -a exp4_disable_args=()
  if [[ "$exp" == "exp4" && "$label" != "full_model" && "$label" != "social_mentalizing" && "$DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING" == "1" ]]; then
    exp4_disable_args=(--disable-exp4-interaction-outcome-pruning)
  fi

  if [[ "$PARALLEL_MULTIAGENT_JOBS" -gt 1 ]]; then
    local -a shard_cmd=(
      bash scripts/utilities/archive/run_reconstruct_sharded.sh
      --exp "$exp"
      --model "$label"
      --steps-file "$steps_file"
      "${replay_trace_args[@]}"
      --inference-file "$inference_file"
      --human-costs-file "$human_costs_file"
      --problem-dir "$problem_dir"
      --move-cost "$MOVE_COST"
      --interact-cost "$INTERACT_COST"
      --observe-cost "$OBSERVE_COST"
      --posterior-candidate-rule "$POSTERIOR_CANDIDATE_RULE"
      --posterior-mass-threshold "$POSTERIOR_MASS_THRESHOLD"
      --posterior-prob-threshold "$POSTERIOR_PROB_THRESHOLD"
      --jobs "$PARALLEL_MULTIAGENT_JOBS"
      --output-file "$output_file"
    )
    if [[ ${#exp4_disable_args[@]} -gt 0 ]]; then
      shard_cmd+=("${exp4_disable_args[@]}")
    fi
    "${shard_cmd[@]}"
  else
    local -a reconstruct_cmd=(
      julia --project=. scripts/utilities/reconstruct_model_costs.jl
      --exp "$exp"
      --model "$label"
      --steps-file "$steps_file"
      "${replay_trace_args[@]}"
      --inference-file "$inference_file"
      --restrict-to-human-levels
      --human-costs-file "$human_costs_file"
      --problem-dir "$problem_dir"
      --move-cost "$MOVE_COST"
      --interact-cost "$INTERACT_COST"
      --observe-cost "$OBSERVE_COST"
      --posterior-candidate-rule "$POSTERIOR_CANDIDATE_RULE"
      --posterior-mass-threshold "$POSTERIOR_MASS_THRESHOLD"
      --posterior-prob-threshold "$POSTERIOR_PROB_THRESHOLD"
      --output-file "$output_file"
    )
    if [[ ${#exp4_disable_args[@]} -gt 0 ]]; then
      reconstruct_cmd+=("${exp4_disable_args[@]}")
    fi
    "${reconstruct_cmd[@]}"
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
echo "Posterior candidate rule: $POSTERIOR_CANDIDATE_RULE (mass threshold=$POSTERIOR_MASS_THRESHOLD, prob threshold=$POSTERIOR_PROB_THRESHOLD)"
echo "Experiments: $EXPERIMENTS"
echo "Parallel multi-agent jobs: $PARALLEL_MULTIAGENT_JOBS"
echo "Disable exp4 interaction-outcome pruning: $DISABLE_EXP4_INTERACTION_OUTCOME_PRUNING"

if has_experiment exp1; then
  run_reconstruct exp1 full_model "$EXP1_MODEL_STEPS" "$EXP1_MODEL_REPLAY_TRACE" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 social_mentalizing "$EXP1_MENTALIZE_STEPS" "$EXP1_MENTALIZE_REPLAY_TRACE" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 rational_non_mentalizing "$EXP1_NONMENTALIZE_STEPS" "$EXP1_NONMENTALIZE_REPLAY_TRACE" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 naive_observer "$EXP1_NAIVE_STEPS" "$EXP1_NAIVE_REPLAY_TRACE" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
  run_reconstruct exp1 agent1_naive_planner "$EXP1_AGENT1_NAIVE_STEPS" "$EXP1_AGENT1_NAIVE_REPLAY_TRACE" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
fi

if has_experiment exp2; then
  run_reconstruct exp2 full_model "$EXP2_MODEL_STEPS" "$EXP2_MODEL_REPLAY_TRACE" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 social_mentalizing "$EXP2_MENTALIZE_STEPS" "$EXP2_MENTALIZE_REPLAY_TRACE" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 rational_non_mentalizing "$EXP2_NONMENTALIZE_STEPS" "$EXP2_NONMENTALIZE_REPLAY_TRACE" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 naive_observer "$EXP2_NAIVE_STEPS" "$EXP2_NAIVE_REPLAY_TRACE" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
  run_reconstruct exp2 agent1_naive_planner "$EXP2_AGENT1_NAIVE_STEPS" "$EXP2_AGENT1_NAIVE_REPLAY_TRACE" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
fi

if has_experiment exp3; then
  run_reconstruct exp3 full_model "$EXP3_MODEL_STEPS" "$EXP3_MODEL_REPLAY_TRACE" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 social_mentalizing "$EXP3_MENTALIZE_STEPS" "$EXP3_MENTALIZE_REPLAY_TRACE" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 rational_non_mentalizing "$EXP3_NONMENTALIZE_STEPS" "$EXP3_NONMENTALIZE_REPLAY_TRACE" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 naive_observer "$EXP3_NAIVE_STEPS" "$EXP3_NAIVE_REPLAY_TRACE" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
  run_reconstruct exp3 agent1_naive_planner "$EXP3_AGENT1_NAIVE_STEPS" "$EXP3_AGENT1_NAIVE_REPLAY_TRACE" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
fi

if has_experiment exp4; then
  run_reconstruct exp4 full_model "$EXP4_MODEL_STEPS" "$EXP4_MODEL_REPLAY_TRACE" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 social_mentalizing "$EXP4_MENTALIZE_STEPS" "$EXP4_MENTALIZE_REPLAY_TRACE" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 rational_non_mentalizing "$EXP4_NONMENTALIZE_STEPS" "$EXP4_NONMENTALIZE_REPLAY_TRACE" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 naive_observer "$EXP4_NAIVE_STEPS" "$EXP4_NAIVE_REPLAY_TRACE" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
  run_reconstruct exp4 agent1_naive_planner "$EXP4_AGENT1_NAIVE_STEPS" "$EXP4_AGENT1_NAIVE_REPLAY_TRACE" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
fi

echo "Done."
