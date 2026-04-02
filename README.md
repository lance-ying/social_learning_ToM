# Social Learning / Bayesian Theory-of-Mind Models

This repository contains a script-driven research codebase for gridworld social-learning experiments. The core model is a Bayesian inverse-planning setup in which an observer reasons about hidden world states, other agents' goals, and whether further observation is worth the cost. Most of the modeling code is in Julia; participant-data processing and figure generation are in Python.

## What the code does

- Represents gridworld tasks as ASCII maps and PDDL planning problems.
- Enumerates hidden belief states, especially uncertainty about which wizard holds the blue key.
- Runs inference over other agents' goals and hidden states using `Gen`, `GenParticleFilters`, `InversePlanning`, `PDDL`, and `SymbolicPlanners`.
- Simulates several observer models, including full/social mentalizing, rational non-mentalizing, and naive baselines.
- Replays model behavior into comparable outputs such as observation counts, replay traces, and reconstructed action costs.
- Compares model predictions against human data and produces analysis tables and plots.

## Repository layout

- `src/`: shared Julia source files.
  - `ascii.jl`: converts ASCII maps into PDDL problems.
  - `beliefs.jl`: enumerates candidate hidden-world states.
  - `planners.jl`: custom planners, including naive wizard-selection behavior.
  - `utils.jl`, `heuristics.jl`, `plan_io.jl`, `render.jl`: planning helpers, caching, replay utilities, and visualization support.
  - `translate.jl`: optional LLM-backed English-to-PDDL belief translation utilities.
- `dataset/`: PDDL domains plus experiment-specific problem sets such as `problems_exp1`, `problems_exp2`, `problems_exp3`, and `problems_exp4_013026`. These folders contain ASCII maps, compiled problems, plans, and metadata.
- `scripts/experiments/`: main experiment runners that generate canonical model outputs under `model_outputs/experiments/`.
  - `run_experiment_exp1.jl`, `run_experiment_exp2.jl`, `run_experiment_exp3.jl`
  - `run_experiment_exp4_wrapper.jl` plus scenario-specific exp4 runners
- `scripts/baselines/`: baseline observer models for each experiment, with outputs written under `model_outputs/baselines/`.
- `scripts/utilities/`: support scripts for building inference files and reconstructing model costs from recorded step/replay traces.
- `inference/`: checked-in `.jld2` posterior files used by the experiment and baseline scripts.
- `model_outputs/`: canonical checked-in outputs for experiments, baselines, and reconstructed costs.
- `step_tables/`: summarized step-count tables by experiment and model.
- `data_processing/`: Python scripts for converting raw study exports, deduplicating participant CSVs, computing statistics, and exporting correlation/cost summaries.
- `plotting/`: Python plotting scripts and generated figures.
- `archive/`: older experiments, notebooks, intermediate data, and deprecated scripts.

## Typical workflow

1. Build or load hidden-state/goal inference data from `scripts/utilities/inference*.jl`, which writes `.jld2` files into `inference/`.
2. Run an experiment script in `scripts/experiments/` or a baseline script in `scripts/baselines/` to produce `steps_dict` and `replay_trace` outputs.
3. Use `scripts/utilities/reconstruct_model_costs.jl` or `scripts/utilities/reconstruct_agent1_naive_planner.jl` to turn those traces into comparable cost summaries in `model_outputs/reconstructed_costs/`.
4. Use the Python scripts in `data_processing/scripts/` and `plotting/` to compare model outputs with human behavior and generate analysis figures.

## Setup

This repo is organized around the root Julia environment in `Project.toml`.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The project targets Julia `1.11`. If `Pkg.instantiate()` cannot resolve `InversePlanning.jl` or `GenGPT3.jl` from your package setup, add them by URL first and then instantiate again.

`OPENAI_API_KEY` is only needed for the optional translation helpers in `src/translate.jl`. The main experiment, baseline, inference, and analysis pipelines do not depend on the API.

## Notes

- This is not a polished Julia package with a single entry-point module; it is a research repository organized around reusable source files plus top-level scripts.
- Many large derived artifacts are intentionally checked in. For current outputs, prefer `model_outputs/` over older files in `scripts/.../experiment_outputs/` or `archive/`.
