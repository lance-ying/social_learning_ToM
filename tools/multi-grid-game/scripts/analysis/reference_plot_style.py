import json

import matplotlib.pyplot as plt
import numpy as np

# HYPERPARAMETERS - Easy to adjust
FIGURE_WIDTH = 11.5
FIGURE_HEIGHT = 2
TITLE_FONTSIZE = 14
YLABEL_FONTSIZE = 12
XLABEL_FONTSIZE = 9
Y_AXIS_MAX = 15
BAR_WIDTH = 0.8
ERROR_BAR_CAPSIZE = 4
DPI = 300
COLORS = ["crimson", "orange"]


with open("/Users/heyodogo/Downloads/plots_social_learning/human_steps.json", "r") as f:
    human_steps = json.load(f)
with open(
    "/Users/heyodogo/Downloads/plots_social_learning/human_steps_std.json", "r"
) as f:
    human_steps_std = json.load(f)
with open(
    "/Users/heyodogo/Downloads/plots_social_learning/step_dict_full.json", "r"
) as f:
    step_dict_full = json.load(f)
with open(
    "/Users/heyodogo/Downloads/plots_social_learning/step_dict_mentalize.json", "r"
) as f:
    step_dict_mentalize = json.load(f)
with open(
    "/Users/heyodogo/Downloads/plots_social_learning/step_dict_nonmentalize.json", "r"
) as f:
    step_dict_nonmentalize = json.load(f)
with open(
    "/Users/heyodogo/Downloads/plots_social_learning/step_dict_naive.json", "r"
) as f:
    step_dict_naive = json.load(f)

scenarios = ["s543_blue_exp", "s543_other_exp"]
scenario_names = ["s543_blue_exp", "s543_other_exp"]
colors = ["crimson", "orange"]
alphas = [0.6, 0.75]
agent_types = [
    "Human",
    "Rational\nMentalizing",
    "Social\nMentalizing",
    "Rational\nNon-mentalizing",
    "Naive",
]

for idx, scenario in enumerate(scenarios):
    plt.figure(figsize=(FIGURE_WIDTH / 2, FIGURE_HEIGHT))

    # Get values: human=human_steps, full=rational mentalizing, mentalize=social mentalizing,
    # nonmentalize=rational non-mentalizing, naive=naive
    values = [
        human_steps.get(scenario, 0),
        step_dict_full.get(scenario, 0),
        step_dict_mentalize.get(scenario, 0),
        step_dict_nonmentalize.get(scenario, 0),
        step_dict_naive.get(scenario, 0),
    ]

    # Create bars
    x_pos = np.arange(len(agent_types))
    plt.bar(x_pos, values, color=colors[idx], width=BAR_WIDTH, alpha=alphas[idx])

    # Add error bar only for human (first bar)
    human_error = human_steps_std.get(scenario, 0)
    plt.errorbar(
        x_pos[0],
        values[0],
        yerr=human_error,
        fmt="none",
        color="black",
        capsize=ERROR_BAR_CAPSIZE,
        capthick=1.5,
        linewidth=1.5,
    )

    plt.xticks(x_pos, agent_types, fontsize=XLABEL_FONTSIZE)
    plt.ylim(0, Y_AXIS_MAX)
    plt.yticks([10])  # Only show tick mark at 10
    plt.ylabel("Number of\nObservations", fontsize=YLABEL_FONTSIZE)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"/Users/heyodogo/Downloads/plots_social_learning/{scenario_names[idx]}_plot.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
