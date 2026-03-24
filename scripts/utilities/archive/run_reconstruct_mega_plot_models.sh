#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-scripts/experiments/experiment_outputs/reconstructed_costs_mega_plot}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"

# Exp1
EXP1_MODEL_STEPS="${EXP1_MODEL_STEPS:-steps.dict.json}"
EXP1_MENTALIZE_STEPS="${EXP1_MENTALIZE_STEPS:-scripts/baselines/exp1/step_dict_mentalize_exp1.json}"
EXP1_NONMENTALIZE_STEPS="${EXP1_NONMENTALIZE_STEPS:-scripts/baselines/exp1/step_dict_nonmentalize_exp1.json}"
EXP1_NAIVE_STEPS="${EXP1_NAIVE_STEPS:-scripts/baselines/exp1/step_dict_naive_exp1.json}"
EXP1_INFERENCE="${EXP1_INFERENCE:-data/inference/inference_data_exp1.jld2}"
EXP1_PROBLEM_DIR="${EXP1_PROBLEM_DIR:-dataset/problems_exp1}"
EXP1_HUMAN_COSTS="${EXP1_HUMAN_COSTS:-data_processing/outputs/human_costs/exp1_human_costs.json}"

# Exp2
EXP2_MODEL_STEPS="${EXP2_MODEL_STEPS:-steps_exp2.json}"
EXP2_MENTALIZE_STEPS="${EXP2_MENTALIZE_STEPS:-scripts/baselines/exp2/step_dict_mentalize_exp2.json}"
EXP2_NONMENTALIZE_STEPS="${EXP2_NONMENTALIZE_STEPS:-scripts/baselines/exp2/step_dict_nonmentalize_exp2.json}"
EXP2_NAIVE_STEPS="${EXP2_NAIVE_STEPS:-scripts/baselines/exp2/step_dict_naive_exp2.json}"
EXP2_INFERENCE="${EXP2_INFERENCE:-data/inference/inference_data_exp2.jld2}"
EXP2_PROBLEM_DIR="${EXP2_PROBLEM_DIR:-dataset/problems_exp2}"
EXP2_HUMAN_COSTS="${EXP2_HUMAN_COSTS:-data_processing/outputs/human_costs/exp2_human_costs.json}"

# Exp3
EXP3_MODEL_STEPS="${EXP3_MODEL_STEPS:-steps_dict_exp3.json}"
EXP3_MENTALIZE_STEPS="${EXP3_MENTALIZE_STEPS:-scripts/baselines/exp3/step_dict_mentalize_exp3.json}"
EXP3_NONMENTALIZE_STEPS="${EXP3_NONMENTALIZE_STEPS:-scripts/baselines/exp3/step_dict_nonmentalize_exp3_v2.json}"
EXP3_NAIVE_STEPS="${EXP3_NAIVE_STEPS:-scripts/baselines/exp3/step_dict_naive_exp3.json}"
EXP3_INFERENCE="${EXP3_INFERENCE:-data/inference/inference_data_exp3.jld2}"
EXP3_PROBLEM_DIR="${EXP3_PROBLEM_DIR:-dataset/problems_exp3}"
EXP3_HUMAN_COSTS="${EXP3_HUMAN_COSTS:-data_processing/outputs/human_costs/exp3_human_costs.json}"

# Exp4
EXP4_MODEL_STEPS="${EXP4_MODEL_STEPS:-scripts/experiments/experiment_outputs/steps_dict_exp4_020126_2.json}"
EXP4_MENTALIZE_STEPS="${EXP4_MENTALIZE_STEPS:-step_dict_mentalize_exp4.json}"
EXP4_NONMENTALIZE_STEPS="${EXP4_NONMENTALIZE_STEPS:-scripts/baselines/exp4/step_dict_nonmentalize_exp4_v2.json}"
EXP4_NAIVE_STEPS="${EXP4_NAIVE_STEPS:-scripts/baselines/exp4/step_dict_naive_exp4.json}"
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

  julia --project=. scripts/utilities/reconstruct_model_costs.jl \
    --exp "$exp" \
    --steps-file "$steps_file" \
    --inference-file "$inference_file" \
    --restrict-to-human-levels \
    --human-costs-file "$human_costs_file" \
    --problem-dir "$problem_dir" \
    --output-file "$output_file"
}

echo "Output directory: $OUTPUT_DIR"

run_reconstruct exp1 full_model "$EXP1_MODEL_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
run_reconstruct exp1 social_mentalizing "$EXP1_MENTALIZE_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
run_reconstruct exp1 rational_non_mentalizing "$EXP1_NONMENTALIZE_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"
run_reconstruct exp1 naive_observer "$EXP1_NAIVE_STEPS" "$EXP1_INFERENCE" "$EXP1_PROBLEM_DIR" "$EXP1_HUMAN_COSTS"

run_reconstruct exp2 full_model "$EXP2_MODEL_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
run_reconstruct exp2 social_mentalizing "$EXP2_MENTALIZE_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
run_reconstruct exp2 rational_non_mentalizing "$EXP2_NONMENTALIZE_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"
run_reconstruct exp2 naive_observer "$EXP2_NAIVE_STEPS" "$EXP2_INFERENCE" "$EXP2_PROBLEM_DIR" "$EXP2_HUMAN_COSTS"

run_reconstruct exp3 full_model "$EXP3_MODEL_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
run_reconstruct exp3 social_mentalizing "$EXP3_MENTALIZE_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
run_reconstruct exp3 rational_non_mentalizing "$EXP3_NONMENTALIZE_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"
run_reconstruct exp3 naive_observer "$EXP3_NAIVE_STEPS" "$EXP3_INFERENCE" "$EXP3_PROBLEM_DIR" "$EXP3_HUMAN_COSTS"

run_reconstruct exp4 full_model "$EXP4_MODEL_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
run_reconstruct exp4 social_mentalizing "$EXP4_MENTALIZE_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
run_reconstruct exp4 rational_non_mentalizing "$EXP4_NONMENTALIZE_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"
run_reconstruct exp4 naive_observer "$EXP4_NAIVE_STEPS" "$EXP4_INFERENCE" "$EXP4_PROBLEM_DIR" "$EXP4_HUMAN_COSTS"

echo "Done."
