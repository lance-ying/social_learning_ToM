#!/bin/bash
# Run experiment for each temperature × threshold combination

MAPS="sm211,sm221,sm341,sm611,sm421"
SCENARIOS="1,2"
TEMPS=(0.3 0.4 0.5)
THRESHOLDS=(0.1 0.15 0.2 0.25)

for temp in "${TEMPS[@]}"; do
    inference_file="inference_exp4_hyperparam_temp${temp}.jld2"

    for thresh in "${THRESHOLDS[@]}"; do
        echo "Running experiment: temp=$temp, threshold=$thresh"
        julia scripts/experiments/run_experiment_exp4_hyperparam.jl \
            "$inference_file" "$thresh" "$MAPS" "$SCENARIOS"
    done
done

echo "Completed 12 experiment runs"
