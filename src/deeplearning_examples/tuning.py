"""Project-independent helpers for persistent Optuna studies."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd


TrialBudgetMode = Literal["additional", "total"]


def require_optuna() -> Any:
    """Import the optional Optuna dependency with a repository-specific hint."""
    try:
        import optuna
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Optuna is required for parameter scans. Install with "
            "`uv sync --extra torch --extra tuning`."
        ) from error
    return optuna


def create_optuna_study(
    *,
    direction: Literal["minimize", "maximize"],
    study_name: str | None,
    storage: str | None,
    sampler_seed: int,
    n_startup_trials: int,
    pruning_warmup_steps: int,
) -> Any:
    """Create or resume a seeded TPE study with median pruning."""
    if n_startup_trials < 0 or pruning_warmup_steps < 0:
        raise ValueError("Optuna startup trials and pruning warmup must be non-negative.")
    optuna = require_optuna()
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=pruning_warmup_steps,
    )
    return optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )


def resolve_optuna_trial_budget(
    study: Any,
    *,
    n_trials: int,
    mode: TrialBudgetMode,
) -> int:
    """Return the number of trials still required by a total/additional budget."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if mode == "additional":
        return n_trials
    if mode == "total":
        return max(0, n_trials - len(study.trials))
    raise ValueError("mode must be 'additional' or 'total'.")


def optuna_parameter_importances(study: Any) -> pd.Series:
    """Return Optuna's global parameter importances as a sorted series."""
    optuna = require_optuna()
    importances = optuna.importance.get_param_importances(study)
    return pd.Series(importances, name="importance", dtype=float).sort_values(
        ascending=False
    )
