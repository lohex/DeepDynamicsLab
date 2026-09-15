"""Controlled ablations and optional Optuna scans for time-course models."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from deeplearning_examples.tuning import (
    TrialBudgetMode,
    create_optuna_study,
    optuna_parameter_importances,
    require_optuna,
    resolve_optuna_trial_budget,
)

from .analysis import classification_metrics, predict_classes
from .data import FoldData, Preprocessor
from .experiments import create_model, fit_model
from .training import EpochMetrics, TrainingConfig, TrainingHistory


ScanScope = Literal["joint", "training", "model"]
ModelConfigSuggester = Callable[[Any, Mapping[str, object]], dict[str, object]]
TrainingConfigSuggester = Callable[[Any, TrainingConfig], TrainingConfig]


EXTENDED_CNN_TRAINING_SEARCH_SPACE: dict[str, object] = {
    "learning_rate": (5e-5, 1e-3),
    "use_weight_decay": (False, True),
    "weight_decay": (1e-6, 1e-3),
    "scheduler": ("none", "step", "cosine", "warmup_cosine"),
    "scheduler_step_size": (8, 12, 16, 20, 24),
    "learning_rate_decay": (0.5, 0.9),
    "warmup_epochs": (3, 5, 10, 15),
    "batch_size": (32, 64, 128, 256),
    "class_balance_strategy": ("none", "weighted_loss", "balanced_sampler"),
    "gradient_clip_norm": (None, 1.0, 5.0),
}

EXTENDED_CNN_COMMON_MODEL_SEARCH_SPACE: dict[str, object] = {
    "head_depth": (1, 2),
    "head_width.branch_models": (64, 128, 256, 512),
    "head_width.specialized_models": (32, 64, 128, 256),
    "dropout.first_difference": (0.05, 0.40),
    "dropout.fft": (0.10, 0.40),
    "dropout.gated_fusion": (0.10, 0.35),
    "dropout.multi_scale": (0.05, 0.50),
    "dropout.tcn": (0.05, 0.45),
    "dropout.shift_robust": (0.05, 0.45),
    "dropout.ordinal": (0.05, 0.40),
}

EXTENDED_CNN_ARCHITECTURE_SEARCH_SPACES: dict[str, dict[str, object]] = {
    "first_difference": {
        "depth": (2, 4), "width_profile": ("narrow", "balanced", "wide"),
        "kernel_profile": ("local", "mixed", "broad", "very_broad"),
        "dilation_profile": ("none", "progressive"),
        "adaptive_pool_size": (1, 2, 4, 8),
    },
    "fft": {
        "depth": (2, 4), "width_profile": ("narrow", "balanced", "wide"),
        "kernel_profile": ("local", "mixed", "broad", "very_broad"),
        "dilation_profile": ("none", "progressive"),
        "adaptive_pool_size": (1, 2, 4, 8),
    },
    "gated_fusion": {
        "depth": (2, 4), "width_profile": ("narrow", "balanced", "wide"),
        "kernel_profile": ("local", "mixed", "broad", "very_broad"),
        "dilation_profile": ("none", "progressive"),
        "adaptive_pool_size": (1, 2, 4, 8),
    },
    "ordinal": {
        "depth": (2, 4), "width_profile": ("narrow", "balanced", "wide"),
        "kernel_profile": ("local", "mixed", "broad", "very_broad"),
        "dilation_profile": ("none", "progressive"),
        "adaptive_pool_size": (1, 2, 4, 8),
        "ordinal_temperature": (0.4, 2.0),
    },
    "multi_scale": {
        "output_channels": (24, 48, 64, 96, 128),
        "multi_scale_profile": ("compact", "balanced", "broad", "very_broad"),
    },
    "tcn": {
        "blocks": (2, 5), "width": (32, 48, 64, 96, 128),
        "kernel_size": (3, 5, 7, 9),
        "dilation_profile": ("none", "progressive"),
    },
    "shift_robust": {
        "depth": (2, 5), "width_profile": ("narrow", "balanced", "wide"),
        "kernel_profile": ("local", "mixed", "broad", "very_broad"),
    },
}


ATTENTION_TRAINING_SEARCH_SPACE: dict[str, object] = {
    "learning_rate": (5e-5, 1e-3),
    "use_weight_decay": (False, True),
    "weight_decay": (1e-6, 2e-3),
    "scheduler": ("none", "step", "cosine", "warmup_cosine"),
    "scheduler_step_size": (8, 12, 16, 20, 24),
    "learning_rate_decay": (0.5, 0.9),
    "warmup_epochs": (3, 5, 10, 15),
    # Full-resolution attention is quadratic in sequence length. These choices
    # avoid carrying the CNN-oriented 256/512 batches into this search.
    "batch_size": (16, 32, 64, 128),
    "class_balance_strategy": ("none", "weighted_loss", "balanced_sampler"),
    "gradient_clip_norm": (1.0, 5.0),
}


ATTENTION_MODEL_SEARCH_SPACES: dict[str, dict[str, object]] = {
    "mean_attention": {
        "embedding_dim": (24, 32, 48, 64, 96, 128),
        "num_heads": (2, 4, 8),
        "num_layers": (1, 5),
        "feedforward_multiplier": (2, 3, 4),
        "dropout": (0.05, 0.40),
    },
    "cls_attention": {
        "embedding_dim": (24, 32, 48, 64, 96, 128),
        "num_heads": (2, 4, 8),
        "num_layers": (1, 5),
        "feedforward_multiplier": (2, 3, 4),
        "dropout": (0.05, 0.50),
    },
    "cnn_attention_pooling": {
        "depth": (2, 4),
        "width_profile": ("narrow", "balanced", "wide"),
        "kernel_profile": ("local", "mixed", "broad"),
        "stride_profile": ("none", "progressive"),
        "attention_hidden_dim": (16, 32, 64, 128),
        "classifier_hidden_dim": (32, 64, 128, 256),
        "dropout": (0.05, 0.50),
    },
    "hierarchical_patch": {
        "patch_size": (4, 8, 16),
        "embedding_dim": (24, 32, 48, 64, 96),
        "num_heads": (2, 4, 8),
        "stage1_layers": (1, 3),
        "stage2_layers": (1, 3),
        "feedforward_multiplier": (2, 3, 4),
        "dropout": (0.05, 0.45),
    },
}


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One named model/training combination evaluated with fixed seeds."""

    name: str
    model_config: Mapping[str, object]
    training_config: TrainingConfig
    random_seeds: tuple[int, ...] = (42, 43, 44)


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Result of one ablation configuration and random seed."""

    name: str
    random_seed: int
    parameter_count: int
    training_loss: float
    tune_loss: float
    training_accuracy: float
    tune_accuracy: float
    training_macro_f1: float
    tune_macro_f1: float
    accuracy_gap: float
    macro_f1_gap: float
    convergence_epoch_95: int | None
    post_best_tune_loss_degradation: float
    tune_loss_roughness: float
    duration_seconds: float
    failed: bool
    failure_reason: str | None
    history: TrainingHistory | None
    model_config: dict[str, object]
    training_config: TrainingConfig


@dataclass(frozen=True, slots=True)
class OptunaScanConfig:
    """Budget and persistence settings for one Optuna study."""

    scope: ScanScope = "joint"
    n_trials: int = 40
    n_trials_mode: TrialBudgetMode = "additional"
    timeout_seconds: float | None = None
    study_name: str | None = None
    storage: str | None = None
    sampler_seed: int = 42
    trial_seed: int = 42
    n_startup_trials: int = 8
    pruning_warmup_epochs: int = 10


def crossed_ablation_specs(
    model_variants: Mapping[str, Mapping[str, object]],
    training_variants: Mapping[str, TrainingConfig],
    *,
    random_seeds: tuple[int, ...] = (42, 43, 44),
) -> tuple[AblationSpec, ...]:
    """Create a balanced model-by-training factorial experiment."""
    if not model_variants or not training_variants:
        raise ValueError("model_variants and training_variants must not be empty.")
    return tuple(
        AblationSpec(
            name=f"model={model_name}__training={training_name}",
            model_config=dict(model_config),
            training_config=training_config,
            random_seeds=random_seeds,
        )
        for model_name, model_config in model_variants.items()
        for training_name, training_config in training_variants.items()
    )


def run_ablations(
    specs: Sequence[AblationSpec],
    *,
    model_type: type[nn.Module],
    training_fold: FoldData,
    tuning_fold: FoldData,
    preprocessor: Preprocessor,
    device: torch.device | str,
) -> tuple[tuple[AblationResult, ...], pd.DataFrame]:
    """Run every ablation with equal seeds and return raw and tabular results."""
    if not specs:
        raise ValueError("specs must contain at least one ablation.")
    results: list[AblationResult] = []
    for spec in specs:
        if not spec.random_seeds:
            raise ValueError(f"Ablation {spec.name!r} has no random seeds.")
        for random_seed in spec.random_seeds:
            training_config = replace(
                spec.training_config,
                random_seed=random_seed,
            )
            model, parameter_count = create_model(
                model_type,
                input_length=training_fold.features.shape[-1],
                num_classes=len(preprocessor.class_names),
                random_seed=random_seed,
                **dict(spec.model_config),
            )
            started_at = perf_counter()
            try:
                history = fit_model(
                    model,
                    training_config,
                    training_fold=training_fold,
                    tuning_fold=tuning_fold,
                    device=device,
                )
            except FloatingPointError as error:
                results.append(
                    AblationResult(
                        name=spec.name,
                        random_seed=random_seed,
                        parameter_count=parameter_count,
                        training_loss=float("nan"),
                        tune_loss=float("nan"),
                        training_accuracy=float("nan"),
                        tune_accuracy=float("nan"),
                        training_macro_f1=float("nan"),
                        tune_macro_f1=float("nan"),
                        accuracy_gap=float("nan"),
                        macro_f1_gap=float("nan"),
                        convergence_epoch_95=None,
                        post_best_tune_loss_degradation=float("nan"),
                        tune_loss_roughness=float("nan"),
                        duration_seconds=perf_counter() - started_at,
                        failed=True,
                        failure_reason=str(error),
                        history=None,
                        model_config=dict(spec.model_config),
                        training_config=training_config,
                    )
                )
                continue
            predictions = predict_classes(model, tuning_fold, device=device)
            metrics = classification_metrics(
                tuning_fold.targets.cpu().numpy(),
                predictions,
            )
            best_index = history.best_epoch - 1
            validation_losses = np.asarray(history.validation_loss)
            roughness = (
                float(np.mean(np.abs(np.diff(validation_losses, n=2))))
                if len(validation_losses) >= 3 else 0.0
            )
            results.append(
                AblationResult(
                    name=spec.name,
                    random_seed=random_seed,
                    parameter_count=parameter_count,
                    training_loss=history.training_loss[best_index],
                    tune_loss=history.validation_loss[best_index],
                    training_accuracy=history.training_accuracy[best_index],
                    tune_accuracy=metrics["tune_accuracy"],
                    training_macro_f1=history.training_macro_f1[best_index],
                    tune_macro_f1=metrics["tune_macro_f1"],
                    accuracy_gap=(
                        history.training_accuracy[best_index]
                        - history.validation_accuracy[best_index]
                    ),
                    macro_f1_gap=(
                        history.training_macro_f1[best_index]
                        - history.validation_macro_f1[best_index]
                    ),
                    convergence_epoch_95=_convergence_epoch_95(validation_losses),
                    post_best_tune_loss_degradation=(
                        history.validation_loss[-1]
                        - history.validation_loss[best_index]
                    ),
                    tune_loss_roughness=roughness,
                    duration_seconds=perf_counter() - started_at,
                    failed=False,
                    failure_reason=None,
                    history=history,
                    model_config=dict(spec.model_config),
                    training_config=training_config,
                )
            )
    resolved_results = tuple(results)
    return resolved_results, ablation_results_frame(resolved_results)


def ablation_results_frame(
    results: Sequence[AblationResult],
) -> pd.DataFrame:
    """Flatten ablation outcomes and configurations into one analysis table."""
    rows: list[dict[str, object]] = []
    for result in results:
        row: dict[str, object] = {
            "name": result.name,
            "random_seed": result.random_seed,
            "parameter_count": result.parameter_count,
            "best_epoch": result.history.best_epoch if result.history else np.nan,
            "epochs_run": len(result.history.training_loss) if result.history else 0,
            "training_loss": result.training_loss,
            "tune_loss": result.tune_loss,
            "training_accuracy": result.training_accuracy,
            "tune_accuracy": result.tune_accuracy,
            "training_macro_f1": result.training_macro_f1,
            "tune_macro_f1": result.tune_macro_f1,
            "accuracy_gap": result.accuracy_gap,
            "macro_f1_gap": result.macro_f1_gap,
            "convergence_epoch_95": result.convergence_epoch_95,
            "post_best_tune_loss_degradation": (
                result.post_best_tune_loss_degradation
            ),
            "tune_loss_roughness": result.tune_loss_roughness,
            "duration_seconds": result.duration_seconds,
            "failed": result.failed,
            "failure_reason": result.failure_reason,
        }
        row.update(
            {
                f"model.{name}": value
                for name, value in result.model_config.items()
            }
        )
        row.update(
            {
                f"training.{name}": value
                for name, value in asdict(result.training_config).items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ablations(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated seeds without hiding run-to-run variability."""
    required = {
        "name",
        "random_seed",
        "tune_loss",
        "tune_accuracy",
        "tune_macro_f1",
    }
    if not required.issubset(results.columns):
        raise ValueError(f"results must contain columns {sorted(required)}.")
    resolved = results.copy()
    optional_defaults: dict[str, object] = {
        "failed": False,
        "tune_macro_f1_delta": np.nan,
        "macro_f1_gap": np.nan,
        "best_epoch": np.nan,
        "convergence_epoch_95": np.nan,
        "post_best_tune_loss_degradation": np.nan,
        "tune_loss_roughness": np.nan,
        "duration_seconds": np.nan,
    }
    for column, default in optional_defaults.items():
        if column not in resolved:
            resolved[column] = default
    return (
        resolved.groupby("name", sort=False)
        .agg(
            runs=("random_seed", "size"),
            failed_runs=("failed", "sum"),
            tune_loss_mean=("tune_loss", "mean"),
            tune_loss_std=("tune_loss", "std"),
            tune_accuracy_mean=("tune_accuracy", "mean"),
            tune_accuracy_std=("tune_accuracy", "std"),
            tune_macro_f1_mean=("tune_macro_f1", "mean"),
            tune_macro_f1_std=("tune_macro_f1", "std"),
            macro_f1_delta_mean=("tune_macro_f1_delta", "mean"),
            macro_f1_delta_std=("tune_macro_f1_delta", "std"),
            macro_f1_gap_mean=("macro_f1_gap", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_std=("best_epoch", "std"),
            convergence_epoch_95_mean=("convergence_epoch_95", "mean"),
            post_best_degradation_mean=(
                "post_best_tune_loss_degradation", "mean"
            ),
            tune_loss_roughness_mean=("tune_loss_roughness", "mean"),
            duration_seconds_mean=("duration_seconds", "mean"),
        )
        .sort_values("tune_macro_f1_mean", ascending=False)
    )


def add_paired_reference_deltas(
    results: pd.DataFrame,
    *,
    reference_name: str,
) -> pd.DataFrame:
    """Add per-seed macro-F1 differences against one shared reference."""
    required = {"name", "random_seed", "tune_macro_f1"}
    if not required.issubset(results.columns):
        raise ValueError(f"results must contain columns {sorted(required)}.")
    reference = results.loc[
        results["name"] == reference_name,
        ["random_seed", "tune_macro_f1"],
    ]
    if reference.empty:
        raise ValueError(f"Reference {reference_name!r} is missing.")
    if reference["random_seed"].duplicated().any():
        raise ValueError("Reference must contain exactly one result per seed.")
    reference = reference.rename(
        columns={"tune_macro_f1": "reference_tune_macro_f1"}
    )
    reference["reference_present"] = True
    paired = results.merge(reference, on="random_seed", how="left", validate="many_to_one")
    if paired["reference_present"].isna().any():
        raise ValueError("Every result seed must have a matching reference run.")
    paired = paired.drop(columns="reference_present")
    paired["tune_macro_f1_delta"] = (
        paired["tune_macro_f1"] - paired["reference_tune_macro_f1"]
    )
    return paired


def save_ablation_study(
    output_dir: str | Path,
    *,
    results: Sequence[AblationResult],
    frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    """Persist tables, configurations and histories, but never model weights."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "runs.csv", index=False)
    summary.to_csv(destination / "summary.csv")
    configurations = [
        {
            "name": result.name,
            "random_seed": result.random_seed,
            "model_config": _jsonable(result.model_config),
            "training_config": _jsonable(asdict(result.training_config)),
            "failed": result.failed,
            "failure_reason": result.failure_reason,
        }
        for result in results
    ]
    with (destination / "configurations.json").open(
        "w", encoding="utf-8"
    ) as configuration_file:
        json.dump(configurations, configuration_file, indent=2)
        configuration_file.write("\n")

    histories: dict[str, np.ndarray] = {}
    for result in results:
        if result.history is None:
            continue
        prefix = re.sub(r"[^A-Za-z0-9_]+", "_", result.name).strip("_")
        prefix = f"{prefix}__seed_{result.random_seed}"
        for field in (
            "training_loss",
            "validation_loss",
            "training_accuracy",
            "validation_accuracy",
            "training_macro_f1",
            "validation_macro_f1",
            "learning_rate",
        ):
            histories[f"{prefix}__{field}"] = np.asarray(
                getattr(result.history, field)
            )
        histories[f"{prefix}__best_epoch"] = np.asarray(
            result.history.best_epoch
        )
    np.savez_compressed(destination / "histories.npz", **histories)
    return destination


def _convergence_epoch_95(validation_losses: np.ndarray) -> int:
    """Return the first epoch reaching 95% of the eventual loss improvement."""
    initial_loss = float(validation_losses[0])
    best_loss = float(np.min(validation_losses))
    target = initial_loss - 0.95 * (initial_loss - best_loss)
    return int(np.flatnonzero(validation_losses <= target)[0] + 1)


def run_optuna_scan(
    *,
    model_type: type[nn.Module],
    base_model_config: Mapping[str, object],
    base_training_config: TrainingConfig,
    training_fold: FoldData,
    tuning_fold: FoldData,
    preprocessor: Preprocessor,
    device: torch.device | str,
    scan_config: OptunaScanConfig | None = None,
    model_config_suggester: ModelConfigSuggester | None = None,
    training_config_suggester: TrainingConfigSuggester | None = None,
    mlflow_tracking: object | None = None,
) -> Any:
    """Optimize CNN and/or training parameters and return an Optuna study.

    Optuna is an optional dependency. Install the project with the ``tuning``
    extra before calling this function.
    """
    optuna = require_optuna()
    cfg = scan_config or OptunaScanConfig()
    suggest_model = model_config_suggester or suggest_cnn_model_config
    suggest_training = training_config_suggester or suggest_training_config
    if cfg.scope not in {"joint", "training", "model"}:
        raise ValueError("scope must be 'joint', 'training' or 'model'.")
    if cfg.n_trials_mode not in {"additional", "total"}:
        raise ValueError("n_trials_mode must be 'additional' or 'total'.")
    if cfg.n_trials < 1 or cfg.n_startup_trials < 0:
        raise ValueError("n_trials must be positive and n_startup_trials non-negative.")

    study = create_optuna_study(
        direction="maximize",
        study_name=cfg.study_name,
        storage=cfg.storage,
        sampler_seed=cfg.sampler_seed,
        n_startup_trials=cfg.n_startup_trials,
        pruning_warmup_steps=cfg.pruning_warmup_epochs,
    )
    trials_to_run = resolve_optuna_trial_budget(
        study,
        n_trials=cfg.n_trials,
        mode=cfg.n_trials_mode,
    )
    tracker = None
    if mlflow_tracking is not None:
        from .tracking import MLflowTracker

        tracker = MLflowTracker(mlflow_tracking)  # type: ignore[arg-type]

    def objective(trial: Any) -> float:
        model_config = dict(base_model_config)
        training_config = replace(
            base_training_config,
            random_seed=cfg.trial_seed,
        )
        if cfg.scope in {"joint", "model"}:
            model_config = suggest_model(trial, model_config)
        if cfg.scope in {"joint", "training"}:
            training_config = suggest_training(trial, training_config)

        model, parameter_count = create_model(
            model_type,
            input_length=training_fold.features.shape[-1],
            num_classes=len(preprocessor.class_names),
            random_seed=cfg.trial_seed,
            **model_config,
        )

        def report_epoch(metrics: EpochMetrics) -> None:
            trial.report(metrics.validation_macro_f1, step=metrics.epoch)
            if tracker is not None:
                tracker.log_epoch(metrics)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_context = (
            tracker.trial_run(
                trial_number=trial.number,
                model_config=model_config,
                training_config=training_config,
                parameter_count=parameter_count,
            )
            if tracker is not None else nullcontext()
        )
        with trial_context:
            history = fit_model(
                model,
                training_config,
                training_fold=training_fold,
                tuning_fold=tuning_fold,
                device=device,
                epoch_callback=report_epoch,
            )
            training_predictions = predict_classes(
                model, training_fold, device=device,
                batch_size=training_config.batch_size,
            )
            training_metrics = classification_metrics(
                training_fold.targets.cpu().numpy(),
                training_predictions,
            )
            predictions = predict_classes(
                model, tuning_fold, device=device,
                batch_size=training_config.batch_size,
            )
            metrics = classification_metrics(
                tuning_fold.targets.cpu().numpy(),
                predictions,
            )
            best_index = history.best_epoch - 1
            trial.set_user_attr("parameter_count", parameter_count)
            trial.set_user_attr("best_epoch", history.best_epoch)
            trial.set_user_attr("tune_loss", history.validation_loss[best_index])
            trial.set_user_attr("tune_accuracy", metrics["tune_accuracy"])
            trial.set_user_attr(
                "training_accuracy", training_metrics["tune_accuracy"]
            )
            trial.set_user_attr(
                "training_macro_f1", training_metrics["tune_macro_f1"]
            )
            trial.set_user_attr(
                "macro_f1_gap",
                training_metrics["tune_macro_f1"] - metrics["tune_macro_f1"],
            )
            trial.set_user_attr("model_config", _jsonable(model_config))
            trial.set_user_attr(
                "training_config",
                _jsonable(asdict(training_config)),
            )
            if tracker is not None:
                tracker.log_trial_summary(
                    {
                        "restored_tune_macro_f1": metrics["tune_macro_f1"],
                        "restored_tune_accuracy": metrics["tune_accuracy"],
                        "restored_tune_loss": history.validation_loss[best_index],
                        "restored_training_macro_f1": training_metrics[
                            "tune_macro_f1"
                        ],
                        "restored_macro_f1_gap": (
                            training_metrics["tune_macro_f1"]
                            - metrics["tune_macro_f1"]
                        ),
                        "best_epoch": history.best_epoch,
                    }
                )
            return metrics["tune_macro_f1"]

    study_context = (
        tracker.study_run(scope=cfg.scope, study_name=study.study_name)
        if tracker is not None else nullcontext()
    )
    with study_context:
        if trials_to_run:
            study.optimize(
                objective,
                n_trials=trials_to_run,
                timeout=cfg.timeout_seconds,
            )
        if tracker is not None and len(study.trials) > 0:
            completed_trials = [
                trial for trial in study.trials
                if trial.state.name == "COMPLETE" and trial.value is not None
            ]
            study_metrics: dict[str, float | int] = {
                "completed_trials": len(completed_trials),
                "pruned_trials": sum(
                    trial.state.name == "PRUNED" for trial in study.trials
                ),
                "failed_trials": sum(
                    trial.state.name == "FAIL" for trial in study.trials
                ),
            }
            if completed_trials:
                study_metrics["best_tune_macro_f1"] = max(
                    float(trial.value) for trial in completed_trials
                )
            tracker.log_trial_summary(study_metrics)
    return study


def suggest_training_config(
    trial: Any,
    base: TrainingConfig,
) -> TrainingConfig:
    """Suggest optimizer, balancing and learning-rate-schedule parameters."""
    learning_rate = trial.suggest_float(
        "training.learning_rate",
        1e-5,
        2e-3,
        log=True,
    )
    use_weight_decay = trial.suggest_categorical(
        "training.use_weight_decay",
        [False, True],
    )
    weight_decay = (
        trial.suggest_float("training.weight_decay", 1e-6, 3e-3, log=True)
        if use_weight_decay
        else 0.0
    )
    scheduler_strategy = trial.suggest_categorical(
        "training.scheduler",
        ["none", "step", "cosine", "warmup_cosine"],
    )
    scheduler_step_size = base.scheduler_step_size
    learning_rate_decay = base.learning_rate_decay
    warmup_epochs = 0
    if scheduler_strategy == "step":
        scheduler_step_size = trial.suggest_int(
            "training.scheduler_step_size",
            8,
            24,
            step=4,
        )
        learning_rate_decay = trial.suggest_float(
            "training.learning_rate_decay",
            0.5,
            0.9,
        )
    elif scheduler_strategy == "warmup_cosine":
        candidates = [value for value in (3, 5, 10, 15) if value < base.epochs]
        warmup_epochs = trial.suggest_categorical(
            "training.warmup_epochs",
            candidates,
        )
    return replace(
        base,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=trial.suggest_categorical(
            "training.batch_size",
            [64, 128, 256, 512],
        ),
        scheduler_strategy=scheduler_strategy,
        scheduler_step_size=scheduler_step_size,
        learning_rate_decay=learning_rate_decay,
        warmup_epochs=warmup_epochs,
        class_balance_strategy=trial.suggest_categorical(
            "training.class_balance_strategy",
            ["none", "weighted_loss", "balanced_sampler"],
        ),
        gradient_clip_norm=trial.suggest_categorical(
            "training.gradient_clip_norm",
            [None, 1.0, 5.0],
        ),
        minimum_learning_rate=min(1e-6, learning_rate / 10),
    )


def suggest_cnn_model_config(
    trial: Any,
    base: Mapping[str, object],
) -> dict[str, object]:
    """Suggest interpretable CNN capacity and receptive-field parameters."""
    depth = trial.suggest_int("model.depth", 2, 4)
    width_name = trial.suggest_categorical(
        "model.width_profile",
        ["narrow", "balanced", "wide"],
    )
    widths = {
        "narrow": (8, 16, 32, 64),
        "balanced": (16, 32, 64, 128),
        "wide": (32, 64, 128, 256),
    }
    kernel_name = trial.suggest_categorical(
        "model.kernel_profile",
        ["local", "mixed", "broad"],
    )
    kernels = {
        "local": (5, 3, 3, 3),
        "mixed": (9, 7, 5, 3),
        "broad": (15, 11, 7, 5),
    }
    dilation_name = trial.suggest_categorical(
        "model.dilation_profile",
        ["none", "progressive"],
    )
    dilations = {
        "none": (1, 1, 1, 1),
        "progressive": (1, 2, 4, 8),
    }
    suggested = dict(base)
    suggested.update(
        channels=widths[width_name][:depth],
        kernel_sizes=kernels[kernel_name][:depth],
        dilations=dilations[dilation_name][:depth],
        pool_size=trial.suggest_categorical("model.pool_size", [2, 3]),
        hidden_dim=trial.suggest_categorical(
            "model.hidden_dim",
            [16, 32, 64, 128, 256],
        ),
        dropout=trial.suggest_float("model.dropout", 0.05, 0.55),
        residual=trial.suggest_categorical("model.residual", [False, True]),
    )
    return suggested


def suggest_extended_cnn_model_config(
    trial: Any,
    base: Mapping[str, object],
    *,
    architectures: Sequence[str] = (
        "first_difference",
        "fft",
        "gated_fusion",
        "multi_scale",
        "tcn",
        "shift_robust",
        "ordinal",
    ),
) -> dict[str, object]:
    """Suggest only parameters that affect one fixed Extended-CNN architecture."""
    supported = set(EXTENDED_CNN_ARCHITECTURE_SEARCH_SPACES)
    choices = tuple(dict.fromkeys(architectures))
    if not choices or not set(choices).issubset(supported):
        raise ValueError(
            "architectures must be a non-empty subset of "
            f"{sorted(supported)}."
        )
    if len(choices) != 1:
        raise ValueError(
            "Extended CNN tuning uses one architecture per Optuna study so that "
            "every architecture receives an explicit trial budget."
        )
    architecture = trial.suggest_categorical("model.architecture", choices)
    space = EXTENDED_CNN_ARCHITECTURE_SEARCH_SPACES[architecture]
    width_profiles = {
        "narrow": (12, 24, 40, 64, 80),
        "balanced": (20, 40, 64, 96, 128),
        "wide": (32, 64, 96, 144, 192),
    }
    kernel_profiles = {
        "local": (5, 3, 3, 3, 3),
        "mixed": (9, 7, 5, 3, 3),
        "broad": (15, 11, 7, 5, 3),
        "very_broad": (21, 15, 11, 7, 5),
    }
    branch_architectures = {"first_difference", "fft", "gated_fusion", "ordinal"}
    head_width_key = (
        "head_width.branch_models"
        if architecture in branch_architectures else "head_width.specialized_models"
    )
    head_width = trial.suggest_categorical(
        "model.head_width",
        EXTENDED_CNN_COMMON_MODEL_SEARCH_SPACE[head_width_key],
    )
    head_depth = trial.suggest_categorical(
        "model.head_depth",
        EXTENDED_CNN_COMMON_MODEL_SEARCH_SPACE["head_depth"],
    )
    hidden_dims = (
        (head_width,)
        if head_depth == 1 else (head_width, max(16, head_width // 4))
    )
    dropout_bounds = EXTENDED_CNN_COMMON_MODEL_SEARCH_SPACE[
        f"dropout.{architecture}"
    ]
    suggested = dict(base)
    suggested.update(
        architecture=architecture,
        hidden_dims=hidden_dims,
        dropout=trial.suggest_float(
            "model.dropout", dropout_bounds[0], dropout_bounds[1]
        ),
    )

    if architecture in branch_architectures:
        depth = trial.suggest_int("model.depth", *space["depth"])
        width_profile = trial.suggest_categorical(
            "model.width_profile", space["width_profile"]
        )
        kernel_profile = trial.suggest_categorical(
            "model.kernel_profile", space["kernel_profile"]
        )
        dilation_profile = trial.suggest_categorical(
            "model.dilation_profile", space["dilation_profile"]
        )
        dilations = (
            (1, 2, 4, 8, 16) if dilation_profile == "progressive"
            else (1, 1, 1, 1, 1)
        )
        suggested.update(
            branch_channels=width_profiles[width_profile][:depth],
            kernel_sizes=kernel_profiles[kernel_profile][:depth],
            dilations=dilations[:depth],
            adaptive_pool_size=trial.suggest_categorical(
                "model.adaptive_pool_size", space["adaptive_pool_size"]
            ),
        )
    elif architecture == "multi_scale":
        output_channels = trial.suggest_categorical(
            "model.output_channels", space["output_channels"]
        )
        scale_profile = trial.suggest_categorical(
            "model.multi_scale_profile", space["multi_scale_profile"]
        )
        suggested.update(
            branch_channels=(output_channels,), kernel_sizes=(3,), dilations=(1,),
            adaptive_pool_size=1,
            multi_scale_kernel_sizes={
                "compact": (3, 7),
                "balanced": (3, 7, 15),
                "broad": (3, 9, 21),
                "very_broad": (3, 11, 25),
            }[scale_profile],
        )
    elif architecture == "tcn":
        blocks = trial.suggest_int("model.tcn_blocks", *space["blocks"])
        width = trial.suggest_categorical("model.tcn_width", space["width"])
        kernel_size = trial.suggest_categorical(
            "model.tcn_kernel_size", space["kernel_size"]
        )
        dilation_profile = trial.suggest_categorical(
            "model.tcn_dilation_profile", space["dilation_profile"]
        )
        dilations = (
            (1, 2, 4, 8, 16) if dilation_profile == "progressive"
            else (1, 1, 1, 1, 1)
        )
        suggested.update(
            branch_channels=(width,) * blocks,
            kernel_sizes=(kernel_size,) * blocks,
            dilations=dilations[:blocks],
            adaptive_pool_size=1,
        )
    else:
        depth = trial.suggest_int("model.shift_depth", *space["depth"])
        width_profile = trial.suggest_categorical(
            "model.shift_width_profile", space["width_profile"]
        )
        kernel_profile = trial.suggest_categorical(
            "model.shift_kernel_profile", space["kernel_profile"]
        )
        suggested.update(
            branch_channels=width_profiles[width_profile][:depth],
            kernel_sizes=kernel_profiles[kernel_profile][:depth],
            dilations=(1,) * depth,
            adaptive_pool_size=1,
        )
    if architecture == "ordinal":
        temperature_bounds = space["ordinal_temperature"]
        suggested["ordinal_temperature"] = trial.suggest_float(
            "model.ordinal_temperature",
            temperature_bounds[0],
            temperature_bounds[1],
            log=True,
        )
    return suggested


def suggest_extended_cnn_training_config(
    trial: Any,
    base: TrainingConfig,
) -> TrainingConfig:
    """Suggest the shared training search space for per-architecture studies."""
    space = EXTENDED_CNN_TRAINING_SEARCH_SPACE
    learning_rate = trial.suggest_float(
        "training.learning_rate", *space["learning_rate"], log=True
    )
    use_weight_decay = trial.suggest_categorical(
        "training.use_weight_decay", space["use_weight_decay"]
    )
    weight_decay = (
        trial.suggest_float(
            "training.weight_decay", *space["weight_decay"], log=True
        )
        if use_weight_decay else 0.0
    )
    scheduler = trial.suggest_categorical(
        "training.scheduler", space["scheduler"]
    )
    step_size = base.scheduler_step_size
    decay = base.learning_rate_decay
    warmup_epochs = 0
    if scheduler == "step":
        step_size = trial.suggest_categorical(
            "training.scheduler_step_size", space["scheduler_step_size"]
        )
        decay = trial.suggest_float(
            "training.learning_rate_decay", *space["learning_rate_decay"]
        )
    elif scheduler == "warmup_cosine":
        warmup_choices = tuple(
            value for value in space["warmup_epochs"] if value < base.epochs
        )
        warmup_epochs = trial.suggest_categorical(
            "training.warmup_epochs", warmup_choices
        )
    return replace(
        base,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=trial.suggest_categorical(
            "training.batch_size", space["batch_size"]
        ),
        scheduler_strategy=scheduler,
        scheduler_step_size=step_size,
        learning_rate_decay=decay,
        warmup_epochs=warmup_epochs,
        class_balance_strategy=trial.suggest_categorical(
            "training.class_balance_strategy", space["class_balance_strategy"]
        ),
        gradient_clip_norm=trial.suggest_categorical(
            "training.gradient_clip_norm", space["gradient_clip_norm"]
        ),
        minimum_learning_rate=min(1e-6, learning_rate / 10),
    )


def extended_cnn_search_space_frame() -> pd.DataFrame:
    """Return the declared Extended-CNN search boundaries for display."""
    rows: list[dict[str, object]] = []
    for parameter, values in EXTENDED_CNN_TRAINING_SEARCH_SPACE.items():
        rows.append({"scope": "training", "parameter": parameter, "values": values})
    for parameter, values in EXTENDED_CNN_COMMON_MODEL_SEARCH_SPACE.items():
        rows.append({"scope": "model.common", "parameter": parameter, "values": values})
    for architecture, parameters in EXTENDED_CNN_ARCHITECTURE_SEARCH_SPACES.items():
        for parameter, values in parameters.items():
            rows.append(
                {"scope": architecture, "parameter": parameter, "values": values}
            )
    return pd.DataFrame(rows).set_index(["scope", "parameter"])


def suggest_attention_model_config(
    trial: Any,
    base: Mapping[str, object],
    *,
    architecture: str,
) -> dict[str, object]:
    """Suggest valid model parameters for one fixed attention architecture."""
    if architecture not in ATTENTION_MODEL_SEARCH_SPACES:
        raise ValueError(
            "architecture must be one of "
            f"{sorted(ATTENTION_MODEL_SEARCH_SPACES)}."
        )
    selected_architecture = trial.suggest_categorical(
        "model.architecture", (architecture,)
    )
    space = ATTENTION_MODEL_SEARCH_SPACES[selected_architecture]
    suggested = dict(base)

    if selected_architecture in {"mean_attention", "cls_attention"}:
        embedding_dim = trial.suggest_categorical(
            "model.embedding_dim", space["embedding_dim"]
        )
        eligible_heads = tuple(
            heads for heads in space["num_heads"]
            if embedding_dim % heads == 0
        )
        feedforward_multiplier = trial.suggest_categorical(
            "model.feedforward_multiplier", space["feedforward_multiplier"]
        )
        suggested.update(
            embedding_dim=embedding_dim,
            num_heads=trial.suggest_categorical(
                "model.num_heads", eligible_heads
            ),
            num_layers=trial.suggest_int(
                "model.num_layers", *space["num_layers"]
            ),
            feedforward_dim=embedding_dim * feedforward_multiplier,
            dropout=trial.suggest_float(
                "model.dropout", *space["dropout"]
            ),
            pooling=(
                "mean" if selected_architecture == "mean_attention" else "cls"
            ),
            norm_first=True,
        )
        return suggested

    if selected_architecture == "cnn_attention_pooling":
        depth = trial.suggest_int("model.depth", *space["depth"])
        width_profile = trial.suggest_categorical(
            "model.width_profile", space["width_profile"]
        )
        kernel_profile = trial.suggest_categorical(
            "model.kernel_profile", space["kernel_profile"]
        )
        stride_profile = trial.suggest_categorical(
            "model.stride_profile", space["stride_profile"]
        )
        channels = {
            "narrow": (16, 32, 48, 64),
            "balanced": (24, 48, 72, 96),
            "wide": (32, 64, 96, 128),
        }[width_profile]
        kernels = {
            "local": (5, 3, 3, 3),
            "mixed": (9, 7, 5, 3),
            "broad": (15, 11, 7, 5),
        }[kernel_profile]
        strides = (
            (2, 2, 1, 1) if stride_profile == "progressive"
            else (1, 1, 1, 1)
        )
        suggested.update(
            channels=channels[:depth],
            kernel_sizes=kernels[:depth],
            strides=strides[:depth],
            attention_hidden_dim=trial.suggest_categorical(
                "model.attention_hidden_dim", space["attention_hidden_dim"]
            ),
            classifier_hidden_dim=trial.suggest_categorical(
                "model.classifier_hidden_dim", space["classifier_hidden_dim"]
            ),
            dropout=trial.suggest_float(
                "model.dropout", *space["dropout"]
            ),
        )
        return suggested

    embedding_dim = trial.suggest_categorical(
        "model.embedding_dim", space["embedding_dim"]
    )
    eligible_heads = tuple(
        heads for heads in space["num_heads"]
        if embedding_dim % heads == 0
    )
    suggested.update(
        patch_size=trial.suggest_categorical(
            "model.patch_size", space["patch_size"]
        ),
        embedding_dim=embedding_dim,
        num_heads=trial.suggest_categorical(
            "model.num_heads", eligible_heads
        ),
        stage1_layers=trial.suggest_int(
            "model.stage1_layers", *space["stage1_layers"]
        ),
        stage2_layers=trial.suggest_int(
            "model.stage2_layers", *space["stage2_layers"]
        ),
        feedforward_multiplier=trial.suggest_categorical(
            "model.feedforward_multiplier", space["feedforward_multiplier"]
        ),
        dropout=trial.suggest_float(
            "model.dropout", *space["dropout"]
        ),
        norm_first=True,
    )
    return suggested


def suggest_attention_training_config(
    trial: Any,
    base: TrainingConfig,
) -> TrainingConfig:
    """Suggest a memory-conscious training space for attention studies."""
    space = ATTENTION_TRAINING_SEARCH_SPACE
    learning_rate = trial.suggest_float(
        "training.learning_rate", *space["learning_rate"], log=True
    )
    use_weight_decay = trial.suggest_categorical(
        "training.use_weight_decay", space["use_weight_decay"]
    )
    weight_decay = (
        trial.suggest_float(
            "training.weight_decay", *space["weight_decay"], log=True
        )
        if use_weight_decay else 0.0
    )
    scheduler = trial.suggest_categorical(
        "training.scheduler", space["scheduler"]
    )
    step_size = base.scheduler_step_size
    decay = base.learning_rate_decay
    warmup_epochs = 0
    if scheduler == "step":
        step_size = trial.suggest_categorical(
            "training.scheduler_step_size", space["scheduler_step_size"]
        )
        decay = trial.suggest_float(
            "training.learning_rate_decay", *space["learning_rate_decay"]
        )
    elif scheduler == "warmup_cosine":
        warmup_choices = tuple(
            value for value in space["warmup_epochs"] if value < base.epochs
        )
        warmup_epochs = trial.suggest_categorical(
            "training.warmup_epochs", warmup_choices
        )
    return replace(
        base,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=trial.suggest_categorical(
            "training.batch_size", space["batch_size"]
        ),
        scheduler_strategy=scheduler,
        scheduler_step_size=step_size,
        learning_rate_decay=decay,
        warmup_epochs=warmup_epochs,
        class_balance_strategy=trial.suggest_categorical(
            "training.class_balance_strategy", space["class_balance_strategy"]
        ),
        gradient_clip_norm=trial.suggest_categorical(
            "training.gradient_clip_norm", space["gradient_clip_norm"]
        ),
        minimum_learning_rate=min(1e-6, learning_rate / 10),
    )


def attention_search_space_frame() -> pd.DataFrame:
    """Return the declared attention search boundaries for display."""
    rows: list[dict[str, object]] = []
    for parameter, values in ATTENTION_TRAINING_SEARCH_SPACE.items():
        rows.append({"scope": "training", "parameter": parameter, "values": values})
    for architecture, parameters in ATTENTION_MODEL_SEARCH_SPACES.items():
        for parameter, values in parameters.items():
            rows.append(
                {"scope": architecture, "parameter": parameter, "values": values}
            )
    return pd.DataFrame(rows).set_index(["scope", "parameter"])


def best_trials_per_architecture(study: Any) -> tuple[Any, ...]:
    """Return the best completed Optuna trial for each sampled architecture."""
    best: dict[str, Any] = {}
    for trial in study.trials:
        if trial.state.name != "COMPLETE" or trial.value is None:
            continue
        architecture = trial.params.get("model.architecture")
        if not isinstance(architecture, str):
            continue
        current = best.get(architecture)
        if current is None or float(trial.value) > float(current.value):
            best[architecture] = trial
    return tuple(sorted(best.values(), key=lambda trial: float(trial.value), reverse=True))


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
