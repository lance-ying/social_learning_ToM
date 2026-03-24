"""
Export correlation results (r, 95% CI, p-value, n) for all experiments to CSV.
Mirrors the computations in correlation_exp{1,2,3,4}_4panel.py.

Output: data_processing/outputs/correlation_results.csv
"""
import csv as csv_mod
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent
data_processing_dir = script_dir.parent
workspace_root = data_processing_dir.parent
output_dir = data_processing_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


def baseline_step_path(exp: str, model: str) -> Path:
    return workspace_root / "scripts" / "baselines" / "outputs" / exp / f"step_dict_{model}.json"

# ── shared stats helpers ───────────────────────────────────────────────────
def _pearson_statistic(a, b, axis=-1):
    """Pearson r that handles both 1-D and 2-D (bootstrap-resampled) inputs."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        return float(stats.pearsonr(a, b)[0])
    a = np.moveaxis(a, axis, -1)
    b = np.moveaxis(b, axis, -1)
    out = np.empty(a.shape[:-1], dtype=float)
    for idx in np.ndindex(a.shape[:-1]):
        out[idx] = float(stats.pearsonr(a[idx], b[idx])[0])
    return out


def compute_stats(x, y, n_resamples=1000, seed=42):
    """Return r, p, ci_low, ci_high for paired arrays x, y."""
    x, y = np.asarray(x), np.asarray(y)
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    try:
        res = stats.bootstrap(
            (x, y), _pearson_statistic,
            paired=True, n_resamples=n_resamples,
            confidence_level=0.95, method="percentile", rng=rng,
        )
    except TypeError:
        res = stats.bootstrap(
            (x, y), _pearson_statistic,
            paired=True, n_resamples=n_resamples,
            confidence_level=0.95, method="percentile", random_state=seed,
        )
    return float(r), float(p), float(res.confidence_interval.low), float(res.confidence_interval.high)


def bootstrap_sd(values, n_boot=1000, seed=42):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    return float(samples.mean(axis=1).std(ddof=1))


# ── CSV accumulator ────────────────────────────────────────────────────────
rows = []

def add_row(exp, agent, model, x, y):
    if len(x) < 3:
        print(f"  Skipped {exp} {agent} {model}: n={len(x)} < 3")
        return
    r, p, ci_low, ci_high = compute_stats(x, y)
    rows.append({
        "experiment": exp,
        "agent": agent,
        "model": model,
        "n": len(x),
        "r": round(r, 4),
        "p": round(p, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    })
    print(f"  {exp} {agent:8s} {model:40s}  r={r:.3f}  p={p:.4f}  CI=[{ci_low:.3f}, {ci_high:.3f}]  n={len(x)}")


# ══════════════════════════════════════════════════════════════════════════
# EXP 1
# ══════════════════════════════════════════════════════════════════════════
print("\n── Exp1 ──────────────────────────────────────────────────────────────")

with open(workspace_root / "model_outputs/experiments/exp1/steps_dict.json") as f:
    steps_dict_exp1 = json.load(f)
with open(baseline_step_path("exp1", "naive_observer")) as f:
    naive_exp1 = json.load(f)
with open(baseline_step_path("exp1", "rational_non_mentalizing")) as f:
    nonmentalize_exp1 = json.load(f)
with open(baseline_step_path("exp1", "social_mentalizing")) as f:
    mentalize_exp1 = json.load(f)

results_dict = {}
exec((data_processing_dir / "results/current/results_dict_exp1_50.py").read_text())
human_exp1 = results_dict.copy()


def collect_exp12(model_dict, human_dict, key_fn):
    x, y = [], []
    for mk in model_dict:
        hk = key_fn(mk)
        if hk in human_dict:
            x.append(model_dict[mk])
            y.append(human_dict[hk]["mean_observe_per_activation"])
    return np.array(x), np.array(y)


main_tfm_exp1     = lambda k: k.replace("_ascii", "_1")
baseline_tfm_exp1 = lambda k: f"mod_{k}_1"

for label, d, tfm in [
    ("Rational Mentalizing (Full Model)", steps_dict_exp1,   main_tfm_exp1),
    ("Social Mentalizing",                mentalize_exp1,     baseline_tfm_exp1),
    ("Rational Non-Mentalizing",          nonmentalize_exp1,  baseline_tfm_exp1),
    ("Naive Observer",                    naive_exp1,         baseline_tfm_exp1),
]:
    x, y = collect_exp12(d, human_exp1, tfm)
    add_row("Exp1", "", label, x, y)


# ══════════════════════════════════════════════════════════════════════════
# EXP 2
# ══════════════════════════════════════════════════════════════════════════
print("\n── Exp2 ──────────────────────────────────────────────────────────────")

with open(workspace_root / "model_outputs/experiments/exp2/steps_dict.json") as f:
    steps_dict_exp2 = json.load(f)
with open(baseline_step_path("exp2", "naive_observer")) as f:
    naive_exp2 = json.load(f)
with open(baseline_step_path("exp2", "rational_non_mentalizing")) as f:
    nonmentalize_exp2 = json.load(f)
with open(baseline_step_path("exp2", "social_mentalizing")) as f:
    mentalize_exp2 = json.load(f)

results_dict = {}
exec((data_processing_dir / "results/current/results_dict_merged_exp2.py").read_text())
human_exp2 = results_dict.copy()

identity_tfm = lambda k: k

for label, d in [
    ("Rational Mentalizing (Full Model)", steps_dict_exp2),
    ("Social Mentalizing",                mentalize_exp2),
    ("Rational Non-Mentalizing",          nonmentalize_exp2),
    ("Naive Observer",                    naive_exp2),
]:
    x, y = collect_exp12(d, human_exp2, identity_tfm)
    add_row("Exp2", "", label, x, y)


# ══════════════════════════════════════════════════════════════════════════
# EXP 3 & EXP 4  (shared CSV-parsing logic)
# ══════════════════════════════════════════════════════════════════════════
LEVEL_RE = re.compile(r"^Level:\s*(.+?)\s*$", re.IGNORECASE)

SKIP_LEVELS = {"comprehension_check", "experiment", "s111_1"}
SKIP_PREFIXES = ("sm111_", "sm112_")


def parse_multi_agent_csv(path):
    """Parse one exp3/exp4 CSV → level -> {agent2_count, agent3_count, activation_count}."""
    level_data = defaultdict(lambda: {"agent2_count": 0, "agent3_count": 0, "activation_count": 0})
    current_level = None
    in_table = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue
            m = LEVEL_RE.match(line)
            if m:
                current_level = m.group(1)
                in_table = False
                continue
            if current_level and line.lower().startswith("timestamp,type"):
                in_table = True
                level_data[current_level]["activation_count"] += 1
                continue
            if current_level and in_table:
                try:
                    import csv as _csv
                    row = next(_csv.reader([line]))
                except Exception:
                    row = line.split(",")
                if len(row) >= 6:
                    type_field = (row[1] or "").strip().upper()
                    if "OBSERVE" in type_field:
                        agent_id = (row[5] or "").strip()
                        if agent_id in ("2.0", "2"):
                            level_data[current_level]["agent2_count"] += 1
                        elif agent_id in ("3.0", "3"):
                            level_data[current_level]["agent3_count"] += 1
    return dict(level_data)


def map_level_name(human_level):
    m = re.match(r"^(sm\d+)_(\d+)$", human_level)
    if m:
        return f"{m.group(1)}_scenario{m.group(2)}"
    m = re.match(r"^(sm\d+)_true$", human_level)
    if m:
        return f"{m.group(1)}_scenario1"
    return human_level


def build_human_stats(csv_dir):
    csv_paths = sorted(Path(csv_dir).glob("*.csv"))
    print(f"  Parsing {len(csv_paths)} CSV files from {csv_dir.name}...")
    per_a2 = defaultdict(list)
    per_a3 = defaultdict(list)
    for p in csv_paths:
        file_data = parse_multi_agent_csv(p)
        for level, counts in file_data.items():
            if level in SKIP_LEVELS or any(level.startswith(pfx) for pfx in SKIP_PREFIXES):
                continue
            model_level = map_level_name(level)
            act = counts["activation_count"]
            if act > 0:
                per_a2[model_level].append(counts["agent2_count"] / act)
                per_a3[model_level].append(counts["agent3_count"] / act)

    human_stats = {}
    for level in set(per_a2) | set(per_a3):
        a2 = per_a2.get(level, [])
        a3 = per_a3.get(level, [])
        human_stats[level] = {
            "agent2_mean": float(np.mean(a2)) if a2 else 0.0,
            "agent3_mean": float(np.mean(a3)) if a3 else 0.0,
        }
    return human_stats


def collect_multi_agent(pred_dict, human_stats, agent):
    count_key = f"{agent}_count"
    mean_key = f"{agent}_mean"
    x, y, seen = [], [], set()
    for model_level in pred_dict:
        if model_level in seen or model_level not in human_stats:
            continue
        x.append(pred_dict[model_level][count_key])
        y.append(human_stats[model_level][mean_key])
        seen.add(model_level)
    return np.array(x), np.array(y)


# ── Exp3 ───────────────────────────────────────────────────────────────────
print("\n── Exp3 ──────────────────────────────────────────────────────────────")

with open(workspace_root / "model_outputs/experiments/exp3/steps_dict.json") as f:
    model_exp3 = json.load(f)
with open(baseline_step_path("exp3", "naive_observer")) as f:
    naive_exp3 = json.load(f)
with open(baseline_step_path("exp3", "rational_non_mentalizing")) as f:
    nonmentalize_exp3 = json.load(f)
with open(baseline_step_path("exp3", "social_mentalizing")) as f:
    mentalize_exp3 = json.load(f)

human_stats_exp3 = build_human_stats(data_processing_dir / "data_processed/exp3")

for agent in ("agent2", "agent3"):
    for label, d in [
        ("Rational Mentalizing (Full Model)", model_exp3),
        ("Social Mentalizing",                mentalize_exp3),
        ("Rational Non-Mentalizing",          nonmentalize_exp3),
        ("Naive Observer",                    naive_exp3),
    ]:
        x, y = collect_multi_agent(d, human_stats_exp3, agent)
        add_row("Exp3", agent, label, x, y)


# ── Exp4 ───────────────────────────────────────────────────────────────────
print("\n── Exp4 ──────────────────────────────────────────────────────────────")

with open(workspace_root / "model_outputs/experiments/exp4/steps_dict.json") as f:
    model_exp4 = json.load(f)
with open(baseline_step_path("exp4", "naive_observer")) as f:
    naive_exp4 = json.load(f)
with open(baseline_step_path("exp4", "rational_non_mentalizing")) as f:
    nonmentalize_exp4 = json.load(f)
with open(baseline_step_path("exp4", "social_mentalizing")) as f:
    mentalize_exp4 = json.load(f)

human_stats_exp4 = build_human_stats(data_processing_dir / "data_processed/exp4")

for agent in ("agent2", "agent3"):
    for label, d in [
        ("Rational Mentalizing (Full Model)", model_exp4),
        ("Social Mentalizing",                mentalize_exp4),
        ("Rational Non-Mentalizing",          nonmentalize_exp4),
        ("Naive Observer",                    naive_exp4),
    ]:
        x, y = collect_multi_agent(d, human_stats_exp4, agent)
        add_row("Exp4", agent, label, x, y)


# ══════════════════════════════════════════════════════════════════════════
# Write CSV
# ══════════════════════════════════════════════════════════════════════════
out_path = output_dir / "correlation_results.csv"
fieldnames = ["experiment", "agent", "model", "n", "r", "p", "ci_low", "ci_high"]

with open(out_path, "w", newline="") as f:
    writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved → {out_path}  ({len(rows)} rows)")
