import json
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("scripts/experiments/experiment_outputs")
TEMPS = [0.3, 0.4, 0.5]
THRESHOLDS = [0.1, 0.15, 0.2, 0.25]
STIMULI = [
    "sm211_scenario1",
    "sm221_scenario2",
    "sm341_scenario1",
    "sm611_scenario2",
    "sm421_scenario2"
]

results = []

for temp in TEMPS:
    for thresh in THRESHOLDS:
        temp_str = str(temp).replace(".", "p")
        thresh_str = str(thresh).replace(".", "p")
        filename = f"steps_dict_hyperparam_temp{temp_str}_thresh{thresh_str}.json"
        filepath = OUTPUT_DIR / filename

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            for stimulus in STIMULI:
                if stimulus in data:
                    results.append({
                        'temperature': temp,
                        'threshold': thresh,
                        'stimulus': stimulus,
                        'agent2_count': data[stimulus]['agent2_count'],
                        'agent3_count': data[stimulus]['agent3_count'],
                        'total_observations': data[stimulus]['agent2_count'] + data[stimulus]['agent3_count']
                    })
        else:
            print(f"Warning: File not found: {filename}")

df = pd.DataFrame(results)
output_path = OUTPUT_DIR / "hyperparam_search_results.csv"
df.to_csv(output_path, index=False)
print(f"Saved results to {output_path}")
print(f"Total rows: {len(df)}")
print(f"\nExpected rows: {len(TEMPS) * len(THRESHOLDS) * len(STIMULI)} (3 temps × 4 thresholds × 5 stimuli)")
print(f"\nPreview of results:")
print(df.head(10))
