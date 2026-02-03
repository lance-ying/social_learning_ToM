#!/bin/bash
# Run inference for each temperature value

MAPS="sm211,sm221,sm341,sm611,sm421"
TEMPS=(0.3 0.4 0.5)

for temp in "${TEMPS[@]}"; do
    output_file="inference_exp4_hyperparam_temp${temp}.jld2"
    echo "Running inference with temperature=$temp"
    julia scripts/utilities/inference_multi_exp4_hyperparam.jl "$MAPS" "$output_file" "$temp"
done

echo "Generated 3 inference files in data/inference/"
