"""Optuna searches for dense temporal burst-segmentation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import pandas as pd
import torch
from torch import Tensor

from deeplearning_examples.tuning import (
    TrialBudgetMode,
    create_optuna_study,
    optuna_parameter_importances,
    require_optuna,
    resolve_optuna_trial_budget,
)

from .data import TrajectoryPreprocessor
from .model import BurstBiGRU, BurstCNN, BurstTCN, BurstUNet, count_parameters, create_burst_model
from .training import (
    EpochMetrics,
    TrainingConfig,
    set_reproducible_seed,
    train_model,
)


@dataclass(frozen=True, slots=True)
class BurstOptunaConfig:
    """Budget, persistence and pruning settings for a burst-model study."""

    n_trials: int = 20
    n_trials_mode: TrialBudgetMode = "total"
    timeout_seconds: float | None = None
    study_name: str = "burst-tcn-joint"
    storage: str | None = None
    sampler_seed: int = 42
    trial_seed: int = 42
    n_startup_trials: int = 5
    pruning_warmup_epochs: int = 8


BurstArchitecture = Literal["tcn", "cnn", "bigru", "unet"]
MODEL_NAMES: dict[BurstArchitecture, str] = {
    "tcn": BurstTCN.model_name,
    "cnn": BurstCNN.model_name,
    "bigru": BurstBiGRU.model_name,
    "unet": BurstUNet.model_name,
}


def run_burst_model_search(
    architecture: BurstArchitecture,
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    base_training_config: TrainingConfig,
    config: BurstOptunaConfig | None = None,
    device: torch.device | str | None = None,
) -> Any:
    """Optimize one dense architecture and optimizer settings on validation AP."""
    optuna = require_optuna()
    cfg = config or BurstOptunaConfig()
    if architecture not in MODEL_NAMES:
        raise ValueError(f"Unknown burst architecture {architecture!r}.")
    if cfg.n_trials < 1 or cfg.n_startup_trials < 0:
        raise ValueError("n_trials must be positive and n_startup_trials non-negative.")
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
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

    def objective(trial: Any) -> float:
        model_config = _suggest_model_config(
            trial,
            architecture=architecture,
            input_channels=training_features.shape[1],
        )
        use_weight_decay = trial.suggest_categorical(
            "training.use_weight_decay", [False, True]
        )
        training_config = replace(
            base_training_config,
            batch_size=trial.suggest_categorical(
                "training.batch_size", [32, 64, 128]
            ),
            learning_rate=trial.suggest_float(
                "training.learning_rate", 1e-4, 3e-3, log=True
            ),
            weight_decay=(
                trial.suggest_float(
                    "training.weight_decay", 1e-6, 1e-3, log=True
                )
                if use_weight_decay
                else 0.0
            ),
            maximum_positive_weight=trial.suggest_categorical(
                "training.maximum_positive_weight", [5.0, 10.0, 20.0]
            ),
            gradient_clip_norm=trial.suggest_categorical(
                "training.gradient_clip_norm", [0.5, 1.0, 2.0]
            ),
            random_seed=cfg.trial_seed,
        )
        set_reproducible_seed(cfg.trial_seed)
        model = create_burst_model(MODEL_NAMES[architecture], model_config)

        def report_epoch(metrics: EpochMetrics) -> None:
            trial.report(metrics.validation_average_precision, step=metrics.epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        history = train_model(
            model,
            training_features,
            training_targets,
            validation_features,
            validation_targets,
            config=training_config,
            device=resolved_device,
            epoch_callback=report_epoch,
        )
        best_index = history.best_epoch - 1
        objective_value = history.validation_average_precision[best_index]
        trial.set_user_attr("best_epoch", history.best_epoch)
        trial.set_user_attr("validation_loss", history.validation_loss[best_index])
        trial.set_user_attr("parameter_count", count_parameters(model))
        trial.set_user_attr("architecture", architecture)
        trial.set_user_attr("model_name", MODEL_NAMES[architecture])
        trial.set_user_attr("model_config", _jsonable(model_config))
        trial.set_user_attr("training_config", _jsonable(asdict(training_config)))
        return float(objective_value)

    if trials_to_run:
        study.optimize(
            objective,
            n_trials=trials_to_run,
            timeout=cfg.timeout_seconds,
        )
    return study


def run_burst_tcn_search(
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    base_training_config: TrainingConfig,
    config: BurstOptunaConfig | None = None,
    device: torch.device | str | None = None,
) -> Any:
    """Backward-compatible TCN-specific wrapper."""
    return run_burst_model_search(
        "tcn",
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        base_training_config=base_training_config,
        config=config,
        device=device,
    )


def burst_trial_frame(
    study: Any,
    *,
    architecture: BurstArchitecture | None = None,
) -> pd.DataFrame:
    """Summarize completed and pruned trials in one flat table.

    ``architecture`` supplies metadata for trials created by the original
    TCN-only implementation, which predates the architecture user attribute.
    """
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        row: dict[str, object] = {
            "trial": trial.number,
            "state": trial.state.name,
            "architecture": trial.user_attrs.get("architecture", architecture),
            "validation_ap": trial.value,
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "validation_loss": trial.user_attrs.get("validation_loss"),
            "parameter_count": trial.user_attrs.get("parameter_count"),
        }
        row.update(trial.params)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["validation_ap", "trial"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def best_burst_model_configs(
    study: Any,
    *,
    architecture: BurstArchitecture | None = None,
    final_epochs: int = 60,
    final_patience: int = 12,
    report_every: int = 5,
) -> tuple[str, dict[str, object], TrainingConfig]:
    """Reconstruct final model/training configurations from the best trial."""
    best_trial = study.best_trial
    model_name_value = best_trial.user_attrs.get("model_name")
    if model_name_value is None:
        if architecture is None:
            stored_config = best_trial.user_attrs.get("model_config", {})
            if isinstance(stored_config, dict) and "dilations" in stored_config:
                architecture = "tcn"
            else:
                raise ValueError(
                    "Legacy trial has no architecture metadata; pass architecture explicitly."
                )
        model_name_value = MODEL_NAMES[architecture]
    model_name = str(model_name_value)
    model_config = dict(best_trial.user_attrs["model_config"])
    if "dilations" in model_config:
        model_config["dilations"] = tuple(model_config["dilations"])
    training_values = dict(best_trial.user_attrs["training_config"])
    training_values.update(
        epochs=final_epochs,
        patience=final_patience,
        report_every=report_every,
    )
    return model_name, model_config, TrainingConfig(**training_values)


def best_burst_tcn_configs(
    study: Any,
    *,
    final_epochs: int = 60,
    final_patience: int = 12,
    report_every: int = 5,
) -> tuple[dict[str, object], TrainingConfig]:
    """Backward-compatible reconstruction for an existing TCN study."""
    model_name, model_config, training_config = best_burst_model_configs(
        study,
        architecture="tcn",
        final_epochs=final_epochs,
        final_patience=final_patience,
        report_every=report_every,
    )
    if model_name != BurstTCN.model_name:
        raise ValueError("The selected study does not contain a BurstTCN.")
    return model_config, training_config


def _suggest_model_config(
    trial: Any,
    *,
    architecture: BurstArchitecture,
    input_channels: int,
) -> dict[str, object]:
    dropout = trial.suggest_float("model.dropout", 0.0, 0.30, step=0.05)
    if architecture == "tcn":
        block_count = trial.suggest_int("model.block_count", 3, 6)
        return {
            "input_channels": input_channels,
            "hidden_channels": trial.suggest_categorical(
                "model.hidden_channels", [16, 32, 48, 64]
            ),
            "kernel_size": trial.suggest_categorical(
                "model.kernel_size", [3, 5, 7]
            ),
            "dilations": tuple(2**index for index in range(block_count)),
            "dropout": dropout,
        }
    if architecture == "cnn":
        return {
            "input_channels": input_channels,
            "hidden_channels": trial.suggest_categorical(
                "model.hidden_channels", [16, 32, 48, 64]
            ),
            "kernel_size": trial.suggest_categorical(
                "model.kernel_size", [3, 5, 9, 15]
            ),
            "depth": trial.suggest_int("model.depth", 2, 6),
            "dropout": dropout,
        }
    if architecture == "unet":
        return {
            "input_channels": input_channels,
            "base_channels": trial.suggest_categorical("model.base_channels", [8, 16, 32]),
            "depth": trial.suggest_int("model.depth", 2, 4),
            "kernel_size": trial.suggest_categorical("model.kernel_size", [3, 5, 7]),
            "dropout": dropout,
        }
    return {
        "input_channels": input_channels,
        "hidden_size": trial.suggest_categorical(
            "model.hidden_size", [16, 32, 48, 64]
        ),
        "num_layers": trial.suggest_int("model.num_layers", 1, 3),
        "dropout": dropout,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "BurstArchitecture",
    "BurstOptunaConfig",
    "best_burst_model_configs",
    "best_burst_tcn_configs",
    "burst_trial_frame",
    "optuna_parameter_importances",
    "run_burst_model_search",
    "run_burst_tcn_search",
]
