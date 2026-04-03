# Plotting Scripts

This directory contains the new model-vs-human plotting scripts for experiments 1 to 4.

Scripts:

- `plot_pooled_observe_steps.py`: pooled observation-step scatterplots across experiments 1 to 4
- `plot_pooled_total_steps.py`: pooled total-step scatterplots across experiments 1 to 4
- `plot_pooled_total_cost.py`: pooled total-cost scatterplot across experiments 1 to 4
- `plot_mean_total_steps_violin.py`: grouped mean total-steps violin plot
- `plot_total_steps_mega.py`: total-steps mega plot across experiments 1 to 4
- `plot_observes_mega.py`: observation-step mega plot with separate exp4 agent 2 and agent 3 rows
- `create_custom_stimuli_plot_pngs.py`: custom-stimuli bar/point plots for selected experiment-level timestamps
- `run_all.py`: runs all six plots

Usage:

```bash
uv run plotting/run_all.py
```

Outputs are written to `plotting/outputs/`. By default, `plot_pooled_total_steps.py` writes both `total_steps_pooled_exp1234.png` and `total_steps_pooled_alternative_models_exp1234.png`. By default, `plot_pooled_observe_steps.py` writes both `observation_steps_model_vs_human_pooled_exp1234.png` and `observation_steps_model_vs_human_pooled_alternative_models_exp1234.png`.

Custom stimuli plot usage:

```bash
uv run plotting/create_custom_stimuli_plot_pngs.py
```

The custom stimuli selection lives in `plotting/custom_stimuli_timestamps.json`.

Both custom stimuli scripts use the shared participant-quality filtering logic from `plotting/common.py` when building human observation summaries.
