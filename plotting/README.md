# Plotting Scripts

This directory contains the new model-vs-human plotting scripts for experiments 1 to 4.

Scripts:

- `plot_pooled_observe_steps.py`: pooled observation-step scatterplot across experiments 1 to 4
- `plot_pooled_total_steps.py`: pooled total-step scatterplot across experiments 1 to 4
- `plot_mean_total_steps_bar.py`: grouped mean total-steps bar plot
- `plot_total_steps_mega.py`: total-steps mega plot across experiments 1 to 4
- `plot_observes_mega.py`: observation-step mega plot with separate exp4 agent 2 and agent 3 rows
- `run_all.py`: runs all five plots

Usage:

```bash
uv run plotting/run_all.py
```

Outputs are written to `plotting/outputs/`.
