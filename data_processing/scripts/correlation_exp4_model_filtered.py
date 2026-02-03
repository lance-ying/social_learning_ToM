#!/usr/bin/env python3
"""
Correlation analysis for EXP4 with model observation filtering.
Excludes stimuli where the model observes more than a specified threshold.
"""
import json
import numpy as np
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt

# Get script directory and workspace root
script_dir = Path(__file__).parent  # data_processing/scripts/
data_processing_dir = script_dir.parent  # data_processing/

def map_level_name(human_level: str) -> str:
    """
    Map human level names (e.g., "sm111_1", "sm541_2") to model level names (e.g., "sm111_scenario1", "sm541_scenario2")
    """
    # Pattern: smXXX_1 -> smXXX_scenario1, smXXX_2 -> smXXX_scenario2
    match = re.match(r'^(sm\d+)_(\d+)$', human_level)
    if match:
        base = match.group(1)
        scenario_num = match.group(2)
        return f"{base}_scenario{scenario_num}"

    # If no match, return as-is (might already be in correct format)
    return human_level


def main(human_json_path: str, model_json_path: str, output_dir: str = None, max_model_obs: int = 10):
    """
    Main correlation analysis function with model observation filtering.

    Args:
        human_json_path: Path to human observation data (from observe_parser_exp4.py)
        model_json_path: Path to model predictions
        output_dir: Optional output directory for plots and results
        max_model_obs: Maximum total model observations allowed (default: 10)
    """
    human_json_path = Path(human_json_path)
    model_json_path = Path(model_json_path)

    if output_dir is None:
        output_dir = data_processing_dir / "outputs/plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model predictions
    with open(model_json_path, 'r') as f:
        model_dict = json.load(f)

    # Load human data
    with open(human_json_path, 'r') as f:
        human_dict = json.load(f)

    print("=" * 80)
    print(f"EXP4: Model Predictions vs Human Data (Model Obs <= {max_model_obs})")
    print("=" * 80)
    print(f"Model levels: {len(model_dict)}")
    print(f"Human levels: {len(human_dict)}")
    print()

    # Collect matching data for agent2 and agent3 separately
    arr_model_agent2 = []
    arr_human_agent2 = []
    arr_model_agent3 = []
    arr_human_agent3 = []
    matched_levels = []
    seen_model_levels = set()  # Track to avoid duplicates
    filtered_out_count = 0

    print("Matching levels:")
    for human_level, human_data in sorted(human_dict.items()):
        # Skip non-game levels
        if human_level in ["comprehension_check", "experiment", "s111_1"]:
            continue

        # Skip sm111 and sm112 levels (tutorial levels)
        if human_level.startswith("sm111_") or human_level.startswith("sm112_"):
            print(f"  ⚠ {human_level} (skipped: tutorial level)")
            continue

        model_level = map_level_name(human_level)

        # Skip if we've already matched this model level (avoid duplicates)
        if model_level in seen_model_levels:
            print(f"  ⚠ {human_level} (duplicate, skipping)")
            continue

        if model_level in model_dict:
            model_agent2 = model_dict[model_level].get("agent2_count", 0)
            model_agent3 = model_dict[model_level].get("agent3_count", 0)
            model_total_obs = model_agent2 + model_agent3

            # Filter out if model observations exceed threshold
            if model_total_obs > max_model_obs:
                print(f"  ⚠ {human_level} → {model_level} (filtered: model obs = {model_total_obs} > {max_model_obs})")
                filtered_out_count += 1
                continue

            # Use mean per activation for human data
            human_total_agent2 = human_data.get("agent2_count", 0)
            human_total_agent3 = human_data.get("agent3_count", 0)
            human_activations = human_data.get("activation_count", 1)
            human_mean_agent2 = human_total_agent2 / human_activations if human_activations > 0 else 0.0
            human_mean_agent3 = human_total_agent3 / human_activations if human_activations > 0 else 0.0

            arr_model_agent2.append(model_agent2)
            arr_human_agent2.append(human_mean_agent2)
            arr_model_agent3.append(model_agent3)
            arr_human_agent3.append(human_mean_agent3)
            matched_levels.append((human_level, model_level))
            seen_model_levels.add(model_level)
            print(f"  ✓ {human_level} → {model_level}: agent2: model={model_agent2}, human={human_mean_agent2:.2f} | agent3: model={model_agent3}, human={human_mean_agent3:.2f} | total={model_total_obs}")
        else:
            print(f"  ⚠ {human_level} → {model_level} (not found in model)")

    print(f"\nMatched {len(arr_model_agent2)} levels")
    print(f"Filtered out {filtered_out_count} levels (model obs > {max_model_obs})")

    # Convert to numpy arrays
    arr_model_agent2 = np.array(arr_model_agent2)
    arr_human_agent2 = np.array(arr_human_agent2)
    arr_model_agent3 = np.array(arr_model_agent3)
    arr_human_agent3 = np.array(arr_human_agent3)

    # Calculate correlations
    if len(arr_model_agent2) > 0:
        corr_agent2 = np.corrcoef(arr_model_agent2, arr_human_agent2)[0, 1] if len(arr_model_agent2) > 1 else np.nan
        corr_agent3 = np.corrcoef(arr_model_agent3, arr_human_agent3)[0, 1] if len(arr_model_agent3) > 1 else np.nan

        print("\n" + "=" * 80)
        print("CORRELATIONS")
        print("=" * 80)
        print(f"Agent 2: r = {corr_agent2:.6f}, N = {len(arr_model_agent2)}")
        print(f"Agent 3: r = {corr_agent3:.6f}, N = {len(arr_model_agent3)}")

        # Save correlation stats to JSON
        correlation_stats = {
            "filter_criterion": f"model_total_obs <= {max_model_obs}",
            "max_model_obs": max_model_obs,
            "filtered_out_count": filtered_out_count,
            "agent2": {
                "r": float(corr_agent2) if not np.isnan(corr_agent2) else None,
                "n": int(len(arr_model_agent2))
            },
            "agent3": {
                "r": float(corr_agent3) if not np.isnan(corr_agent3) else None,
                "n": int(len(arr_model_agent3))
            }
        }

        stats_path = output_dir / "correlation_stats_exp4_model_filtered.json"
        with open(stats_path, 'w') as f:
            json.dump(correlation_stats, f, indent=2)
        print(f"\n✓ Saved correlation stats to {stats_path}")

        # Create Agent 2 scatter plot (matching exp3 style)
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(arr_model_agent2, arr_human_agent2, alpha=0.7)
        ax1.set_xlabel('Model Agent 2 Count', fontsize=18)
        ax1.set_ylabel('Human Mean Agent 2', fontsize=18)
        ax1.set_title(f'exp4 agent2 corr (model obs ≤ {max_model_obs})\nr = {corr_agent2:.3f}', fontsize=18)
        ax1.grid(False)

        # Add line of best fit for Agent 2
        if len(arr_model_agent2) > 1 and not np.all(arr_model_agent2 == arr_model_agent2[0]):
            fit2 = np.polyfit(arr_model_agent2, arr_human_agent2, 1)
            fit_fn2 = np.poly1d(fit2)
            x_vals2 = np.linspace(min(arr_model_agent2), max(arr_model_agent2), 100)
            ax1.plot(x_vals2, fit_fn2(x_vals2), color='red', linestyle='-', linewidth=2)

        plt.tight_layout()
        plot_path_agent2 = output_dir / f'correlation_scatterplot_agent2_model_filtered_{max_model_obs}.png'
        plt.savefig(plot_path_agent2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Agent 2 correlation scatterplot to {plot_path_agent2}")

        # Create Agent 3 scatter plot (matching exp3 style)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(arr_model_agent3, arr_human_agent3, alpha=0.7)
        ax2.set_xlabel('Model Agent 3 Count', fontsize=18)
        ax2.set_ylabel('Human Mean Agent 3', fontsize=18)
        ax2.set_title(f'exp4 agent3 corr (model obs ≤ {max_model_obs})\nr = {corr_agent3:.3f}', fontsize=18)
        ax2.grid(False)

        # Add line of best fit for Agent 3
        if len(arr_model_agent3) > 1 and not np.all(arr_model_agent3 == arr_model_agent3[0]):
            fit3 = np.polyfit(arr_model_agent3, arr_human_agent3, 1)
            fit_fn3 = np.poly1d(fit3)
            x_vals3 = np.linspace(min(arr_model_agent3), max(arr_model_agent3), 100)
            ax2.plot(x_vals3, fit_fn3(x_vals3), color='red', linestyle='-', linewidth=2)

        plt.tight_layout()
        plot_path_agent3 = output_dir / f'correlation_scatterplot_agent3_model_filtered_{max_model_obs}.png'
        plt.savefig(plot_path_agent3, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Agent 3 correlation scatterplot to {plot_path_agent3}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return correlation_stats
    else:
        print("\n⚠ No matching levels found for correlation analysis")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python correlation_exp4_model_filtered.py <human_json> <model_json> [output_dir] [max_model_obs]")
        print("\nExample:")
        print("  python correlation_exp4_model_filtered.py \\")
        print("    analysis_results/overall_observations.json \\")
        print("    results/dictionaries/steps_dict_exp4_point5_updated.json \\")
        print("    outputs/plots \\")
        print("    10")
        print("\nDefaults:")
        print("  max_model_obs: 10")
        sys.exit(1)

    human_file = sys.argv[1]
    model_file = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else None
    max_obs = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    main(human_file, model_file, output_directory, max_obs)
