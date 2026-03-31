from __future__ import annotations

from pathlib import Path


def preferred_model_name(exp: str, model_name: str) -> str:
    if model_name == "social_mentalizing" and exp in {"exp3", "exp4"}:
        return "social_mentalizing_until_one_converges"
    if model_name == "social_mentalizing_until_one_converges" and exp in {"exp1", "exp2"}:
        return "social_mentalizing"
    return model_name


def reconstructed_costs_path(repo_root: Path, exp: str, model_name: str) -> Path:
    resolved_model_name = preferred_model_name(exp, model_name)
    default_dir = repo_root / "model_outputs" / "reconstructed_costs"
    if default_dir.exists():
        return default_dir / f"{exp}_{resolved_model_name}.json"
    legacy_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"
    return legacy_dir / f"{exp}_{resolved_model_name}.json"


def baseline_step_path(repo_root: Path, exp: str, model_name: str) -> Path:
    resolved_model_name = preferred_model_name(exp, model_name)
    primary_path = repo_root / "model_outputs" / "baselines" / exp / f"step_dict_{resolved_model_name}.json"
    if primary_path.exists():
        return primary_path
    legacy_path = repo_root / "scripts" / "baselines" / "outputs" / exp / f"step_dict_{resolved_model_name}.json"
    if legacy_path.exists():
        return legacy_path
    return primary_path
