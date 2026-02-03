# Hyperparameter Search for Exp4

## Overview
This directory contains scripts for running hyperparameter search on 5 specific stimuli.

## Hyperparameters
- **Temperature (action noise):** [0.3, 0.4, 0.5]
- **Threshold:** [0.1, 0.15, 0.2, 0.25]

## Stimuli
- sm211_scenario1
- sm221_scenario2
- sm341_scenario1
- sm611_scenario2
- sm421_scenario2

## Total Runs
- **Inference runs:** 3 (one per temperature)
- **Experiment runs:** 12 (3 inference files × 4 thresholds)

## Architecture

### Two-Stage Approach
1. **Stage 1 (Inference):** Generate 3 inference files (one per temperature)
2. **Stage 2 (Experiment):** Run each inference file with 4 different thresholds

## Usage

### Step 1: Run inference sweep (generates 3 inference files)
```bash
bash scripts/hyperparam/run_inference_sweep.sh
```

This will create:
- `data/inference/inference_exp4_hyperparam_temp0.3.jld2`
- `data/inference/inference_exp4_hyperparam_temp0.4.jld2`
- `data/inference/inference_exp4_hyperparam_temp0.5.jld2`

**Estimated time:** ~2-3 hours total

### Step 2: Run experiment sweep (12 runs)
```bash
bash scripts/hyperparam/run_threshold_sweep.sh
```

This will create 12 JSON files in `scripts/experiments/experiment_outputs/`:
- `steps_dict_hyperparam_temp0p3_thresh0p1.json`
- `steps_dict_hyperparam_temp0p3_thresh0p15.json`
- ... (and so on for all combinations)

**Estimated time:** ~1-2 hours total

### Step 3: Aggregate results
```bash
python scripts/hyperparam/aggregate_hyperparam_results.py
```

This creates `scripts/experiments/experiment_outputs/hyperparam_search_results.csv` with 60 rows (12 runs × 5 stimuli).

### Step 4: Analyze results
Results are saved to `scripts/experiments/experiment_outputs/hyperparam_search_results.csv`

Example analysis:
```python
import pandas as pd
df = pd.read_csv("scripts/experiments/experiment_outputs/hyperparam_search_results.csv")
print(df.head())
print(f"Unique temperatures: {df['temperature'].unique()}")
print(f"Unique thresholds: {df['threshold'].unique()}")
print(f"Unique stimuli: {df['stimulus'].unique()}")
```

## Files Created/Modified

### New Julia Scripts
1. `scripts/utilities/inference_multi_exp4_hyperparam.jl` - Modified inference with temperature parameter
2. `scripts/experiments/run_experiment_exp4_hyperparam.jl` - Modified experiment with threshold parameter

### Wrapper Scripts
3. `scripts/hyperparam/run_inference_sweep.sh` - Runs all 3 inference variations
4. `scripts/hyperparam/run_threshold_sweep.sh` - Runs all 12 experiment variations

### Analysis Scripts
5. `scripts/hyperparam/aggregate_hyperparam_results.py` - Aggregates results into CSV

## Expected Runtime
- **Inference:** ~30-60 min per temperature (3 total = ~2-3 hours)
- **Experiment:** ~5-10 min per run (12 total = ~1-2 hours)
- **Total:** ~3-5 hours for complete sweep

## Notes
- Original scripts remain untouched (`inference_multi_exp4.jl`, `run_experiment_exp4.jl`)
- All new scripts use `_hyperparam` suffix to distinguish from originals
- Results can be compared against human data using existing correlation scripts
- Debug logs are saved with hyperparameter info in filenames for easy tracking

## Command-Line Arguments

### Inference Script (`inference_multi_exp4_hyperparam.jl`)
```bash
julia scripts/utilities/inference_multi_exp4_hyperparam.jl <maps> <output_file> <temperature>
```
- `maps`: Comma-separated map IDs (e.g., "sm211,sm221,sm341,sm611,sm421")
- `output_file`: Output filename (e.g., "inference_exp4_hyperparam_temp0.3.jld2")
- `temperature`: Temperature value (e.g., "0.3", "0.4", "0.5")

### Experiment Script (`run_experiment_exp4_hyperparam.jl`)
```bash
julia scripts/experiments/run_experiment_exp4_hyperparam.jl <inference_file> <threshold> <maps> <scenarios>
```
- `inference_file`: Inference data filename (e.g., "inference_exp4_hyperparam_temp0.3.jld2")
- `threshold`: Threshold value (e.g., "0.1", "0.15", "0.2", "0.25")
- `maps`: Comma-separated map IDs (e.g., "sm211,sm221,sm341,sm611,sm421")
- `scenarios`: Comma-separated scenario IDs (e.g., "1,2")

## Verification Steps

After running the sweep, verify:

1. **Inference files exist:**
   ```bash
   ls data/inference/inference_exp4_hyperparam_temp*.jld2
   ```
   Should show 3 files.

2. **Experiment files exist:**
   ```bash
   ls scripts/experiments/experiment_outputs/steps_dict_hyperparam_*.json
   ```
   Should show 12 files.

3. **Aggregated results:**
   ```bash
   wc -l scripts/experiments/experiment_outputs/hyperparam_search_results.csv
   ```
   Should show 61 lines (60 data rows + 1 header).
