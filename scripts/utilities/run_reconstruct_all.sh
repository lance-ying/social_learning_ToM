#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-scripts/experiments/experiment_outputs}"
EXP3_STEPS="${EXP3_STEPS:-scripts/experiments/experiment_outputs/steps_dict_exp3_test.json}"
EXP4_STEPS="${EXP4_STEPS:-scripts/experiments/experiment_outputs/steps_dict_exp4_020126_1.json}"
EXP4_INFERENCE="${EXP4_INFERENCE:-data/inference/inference_exp4_020126_1.jld2}"
EXP4_PROBLEM_DIR="${EXP4_PROBLEM_DIR:-dataset/problems_exp4}"

echo "[1/4] reconstruct exp1"
julia --project=. scripts/utilities/reconstruct_model_costs.jl \
  --exp exp1 \
  --steps-file scripts/experiments/experiment_outputs/steps_dict_exp1.json \
  --output-file "$OUTPUT_DIR/reconstructed_costs_exp1.json"

echo "[2/4] reconstruct exp2"
julia --project=. scripts/utilities/reconstruct_model_costs.jl \
  --exp exp2 \
  --steps-file scripts/experiments/experiment_outputs/steps_dict_exp2.json \
  --output-file "$OUTPUT_DIR/reconstructed_costs_exp2.json"

echo "[3/4] reconstruct exp3_debug"
julia --project=. scripts/utilities/reconstruct_model_costs.jl \
  --exp exp3_debug \
  --steps-file "$EXP3_STEPS" \
  --output-file "$OUTPUT_DIR/reconstructed_costs_exp3_debug.json"

echo "[4/4] reconstruct exp4_wrapper"
julia --project=. scripts/utilities/reconstruct_model_costs.jl \
  --exp exp4_wrapper \
  --steps-file "$EXP4_STEPS" \
  --inference-file "$EXP4_INFERENCE" \
  --problem-dir "$EXP4_PROBLEM_DIR" \
  --output-file "$OUTPUT_DIR/reconstructed_costs_exp4_wrapper.json"

echo "done"
