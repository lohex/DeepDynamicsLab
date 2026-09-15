"""Controlled data/model scaling experiments for the baseline time-course CNN."""

from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
from json import dumps
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from torch import Tensor, nn

from .data import FoldData, Preprocessor, subset_fold
from .experiments import create_model, fit_model
from .models.cnn import scaled_cnn_config
from .training import TrainingConfig


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    """Grid definition for the controlled N x P experiment."""

    data_fractions: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    width_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    learning_rates: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    subset_seed: int = 42
    model_seed: int = 42

    def __post_init__(self) -> None:
        _validate_fractions(self.data_fractions)
        _validate_positive_unique(self.width_multipliers, "width_multipliers")
        _validate_positive_unique(self.learning_rates, "learning_rates")


@dataclass(frozen=True, slots=True)
class MemorizationResult:
    """Outcome of the deliberately tiny-data trainability check."""

    n_samples: int
    parameter_count: int
    initial_loss: float
    final_loss: float
    final_accuracy: float
    best_epoch: int
    validation_best_epoch: int | None
    passed: bool
    loss_curve: tuple[float, ...]
    accuracy_curve: tuple[float, ...]
    validation_loss_curve: tuple[float, ...]
    validation_accuracy_curve: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScalingSurfaceFit:
    """Parameters of L(N, P) = L_inf + A N^-alpha + B P^-beta."""

    irreducible_loss: float
    data_coefficient: float
    data_exponent: float
    parameter_coefficient: float
    parameter_exponent: float


def nested_stratified_indices(
    targets: Tensor,
    fractions: Sequence[float],
    *,
    random_seed: int = 42,
) -> dict[float, np.ndarray]:
    """Return class-stratified subsets nested as data size grows."""
    resolved_fractions = tuple(float(value) for value in fractions)
    _validate_fractions(resolved_fractions)
    target_values = _target_array(targets)
    rng = np.random.default_rng(random_seed)
    classes = np.unique(target_values)
    pools = {
        int(label): rng.permutation(np.flatnonzero(target_values == label))
        for label in classes
    }

    subsets: dict[float, np.ndarray] = {}
    previous: set[int] = set()
    for fraction in resolved_fractions:
        selected = []
        for label in classes:
            pool = pools[int(label)]
            count = len(pool) if fraction == 1.0 else max(1, round(len(pool) * fraction))
            selected.append(pool[: min(count, len(pool))])
        indices = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
        current = set(indices.tolist())
        if not previous.issubset(current):
            raise RuntimeError("Generated scaling subsets are not nested.")
        subsets[fraction] = indices
        previous = current
    return subsets


def make_scaling_folds(
    training_fold: FoldData,
    fractions: Sequence[float],
    *,
    random_seed: int = 42,
) -> dict[float, FoldData]:
    """Construct the nested training folds used by the scaling grid."""
    indices = nested_stratified_indices(
        training_fold.targets,
        fractions,
        random_seed=random_seed,
    )
    return {
        fraction: subset_fold(training_fold, subset_indices)
        for fraction, subset_indices in indices.items()
    }


def run_memorization_check(
    *,
    model_type: type[nn.Module],
    training_fold: FoldData,
    validation_fold: FoldData | None = None,
    preprocessor: Preprocessor,
    base_model_config: Mapping[str, object],
    width_multiplier: float = 1.0,
    sample_count: int = 32,
    epochs: int = 200,
    learning_rate: float = 3e-3,
    random_seed: int = 42,
    device: torch.device | str = "cpu",
    accuracy_threshold: float = 0.99,
) -> MemorizationResult:
    """Verify that the model/pipeline can memorize one stratified mini-dataset."""
    if sample_count < len(preprocessor.class_names):
        raise ValueError("sample_count must be at least the number of classes.")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")
    if not 0 < accuracy_threshold <= 1:
        raise ValueError("accuracy_threshold must be in (0, 1].")

    indices = _balanced_sample_indices(
        training_fold.targets,
        min(sample_count, len(training_fold.targets)),
        random_seed=random_seed,
    )
    small_fold = subset_fold(training_fold, indices)
    model_config = dict(base_model_config)
    model_config.update(scaled_cnn_config(width_multiplier))
    model_config["dropout"] = 0.0
    model, parameter_count = create_model(
        model_type,
        input_length=small_fold.features.shape[-1],
        num_classes=len(preprocessor.class_names),
        random_seed=random_seed,
        **model_config,
    )

    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    with torch.no_grad():
        logits = model(small_fold.features.to(resolved_device))
        initial_loss = float(
            nn.functional.cross_entropy(
                logits,
                small_fold.targets.to(resolved_device),
            ).item()
        )

    history = fit_model(
        model,
        TrainingConfig(
            epochs=epochs,
            batch_size=len(small_fold.targets),
            learning_rate=learning_rate,
            weight_decay=0.0,
            scheduler_strategy="none",
            patience=epochs,
            report_every=max(1, epochs // 5),
            random_seed=random_seed,
            class_balance_strategy="none",
            gradient_clip_norm=None,
        ),
        training_fold=small_fold,
        tuning_fold=validation_fold or small_fold,
        device=resolved_device,
    )
    if validation_fold is None:
        loss_curve = history.validation_loss
        accuracy_curve = history.validation_accuracy
        validation_loss_curve: tuple[float, ...] = ()
        validation_accuracy_curve: tuple[float, ...] = ()
        validation_best_epoch = None
        best_index = history.best_epoch - 1
    else:
        loss_curve = history.training_loss
        accuracy_curve = history.training_accuracy
        validation_loss_curve = history.validation_loss
        validation_accuracy_curve = history.validation_accuracy
        validation_best_epoch = history.best_epoch
        best_index = int(np.argmin(loss_curve))
    final_accuracy = float(accuracy_curve[best_index])
    return MemorizationResult(
        n_samples=len(small_fold.targets),
        parameter_count=parameter_count,
        initial_loss=initial_loss,
        final_loss=float(loss_curve[best_index]),
        final_accuracy=final_accuracy,
        best_epoch=best_index + 1,
        validation_best_epoch=validation_best_epoch,
        passed=final_accuracy >= accuracy_threshold,
        loss_curve=loss_curve,
        accuracy_curve=accuracy_curve,
        validation_loss_curve=validation_loss_curve,
        validation_accuracy_curve=validation_accuracy_curve,
    )


def run_scaling_grid(
    *,
    model_type: type[nn.Module],
    training_fold: FoldData,
    tuning_fold: FoldData,
    preprocessor: Preprocessor,
    base_model_config: Mapping[str, object],
    base_training_config: TrainingConfig,
    scaling_config: ScalingConfig,
    device: torch.device | str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run nested N x P experiments with an inner learning-rate sweep."""
    resolved_device = torch.device(device)
    folds = make_scaling_folds(
        training_fold,
        scaling_config.data_fractions,
        random_seed=scaling_config.subset_seed,
    )
    rows: list[dict[str, object]] = []

    for fraction in scaling_config.data_fractions:
        scaled_fold = folds[fraction]
        for width in scaling_config.width_multipliers:
            model_config = dict(base_model_config)
            model_config.update(scaled_cnn_config(width))
            for learning_rate in scaling_config.learning_rates:
                row = _run_scaling_cell(
                    model_type=model_type,
                    training_fold=scaled_fold,
                    tuning_fold=tuning_fold,
                    num_classes=len(preprocessor.class_names),
                    model_config=model_config,
                    training_config=replace(
                        base_training_config,
                        learning_rate=learning_rate,
                        random_seed=scaling_config.model_seed,
                    ),
                    model_seed=scaling_config.model_seed,
                    device=resolved_device,
                )
                row.update(
                    data_fraction=fraction,
                    width_multiplier=width,
                    learning_rate=learning_rate,
                )
                rows.append(row)
                if verbose:
                    _print_scaling_row(row)
    return pd.DataFrame(rows)


def select_best_learning_rate(runs: pd.DataFrame) -> pd.DataFrame:
    """Select minimum tune loss independently for every N x P cell."""
    required = {
        "data_fraction", "n_samples", "width_multiplier", "parameter_count",
        "learning_rate", "tune_loss", "failed",
    }
    _require_columns(runs, required, "runs")
    successful = runs.loc[
        (~runs["failed"].astype(bool)) & np.isfinite(runs["tune_loss"])
    ]
    if successful.empty:
        raise ValueError("No successful scaling runs are available.")
    groups = ["data_fraction", "n_samples", "width_multiplier", "parameter_count"]
    best_indices = successful.groupby(groups, sort=False)["tune_loss"].idxmin()
    return successful.loc[best_indices].sort_values(groups).reset_index(drop=True)


def near_optimal_parameter_count(
    selected_runs: pd.DataFrame,
    *,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """Return the smallest P within an absolute tune-loss tolerance of the best P."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    required = {"n_samples", "parameter_count", "width_multiplier", "tune_loss"}
    _require_columns(selected_runs, required, "selected_runs")
    rows = []
    for n_samples, group in selected_runs.groupby("n_samples", sort=True):
        group = group.sort_values("parameter_count")
        best_loss = float(group["tune_loss"].min())
        threshold = best_loss + tolerance
        chosen = group.loc[group["tune_loss"] <= threshold].iloc[0]
        rows.append(
            {
                "n_samples": int(n_samples),
                "best_tune_loss": best_loss,
                "loss_threshold": threshold,
                "parameter_count": int(chosen["parameter_count"]),
                "width_multiplier": float(chosen["width_multiplier"]),
                "selected_tune_loss": float(chosen["tune_loss"]),
            }
        )
    return pd.DataFrame(rows)


def compute_efficiency_frontier(
    selected_runs: pd.DataFrame,
    *,
    cost_column: str = "duration_seconds",
    loss_column: str = "tune_loss",
) -> pd.DataFrame:
    """Return runs that improve best observed loss as compute cost increases."""
    _require_columns(selected_runs, {cost_column, loss_column}, "selected_runs")
    ordered = selected_runs.loc[
        np.isfinite(selected_runs[cost_column]) & np.isfinite(selected_runs[loss_column])
    ].sort_values(cost_column)
    if ordered.empty:
        raise ValueError("No finite compute-efficiency runs are available.")
    best_loss = float("inf")
    frontier = []
    for index, row in ordered.iterrows():
        loss = float(row[loss_column])
        if loss < best_loss:
            frontier.append(index)
            best_loss = loss
    return ordered.loc[frontier].reset_index(drop=True)


def fit_scaling_surface(selected_runs: pd.DataFrame) -> ScalingSurfaceFit:
    """Fit a descriptive power-law surface to tune cross-entropy."""
    _require_columns(
        selected_runs,
        {"n_samples", "parameter_count", "tune_loss"},
        "selected_runs",
    )
    data = selected_runs.loc[
        np.isfinite(selected_runs["tune_loss"])
        & (selected_runs["n_samples"] > 0)
        & (selected_runs["parameter_count"] > 0)
    ]
    if len(data) < 5:
        raise ValueError("At least five successful N x P cells are required.")

    n_values = data["n_samples"].to_numpy(dtype=float)
    p_values = data["parameter_count"].to_numpy(dtype=float)
    losses = data["tune_loss"].to_numpy(dtype=float)
    fitted, _ = curve_fit(
        _scaling_surface,
        (n_values, p_values),
        losses,
        p0=(max(0.0, float(losses.min()) * 0.8), 1.0, 0.5, 1.0, 0.5),
        bounds=(
            (0.0, 0.0, 1e-6, 0.0, 1e-6),
            (np.inf, np.inf, 5.0, np.inf, 5.0),
        ),
        maxfev=50_000,
    )
    return ScalingSurfaceFit(*map(float, fitted))


def _run_scaling_cell(
    *,
    model_type: type[nn.Module],
    training_fold: FoldData,
    tuning_fold: FoldData,
    num_classes: int,
    model_config: Mapping[str, object],
    training_config: TrainingConfig,
    model_seed: int,
    device: torch.device,
) -> dict[str, object]:
    with redirect_stdout(StringIO()):
        model, parameter_count = create_model(
            model_type,
            input_length=training_fold.features.shape[-1],
            num_classes=num_classes,
            random_seed=model_seed,
            **model_config,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    try:
        with redirect_stdout(StringIO()):
            history = fit_model(
                model,
                training_config,
                training_fold=training_fold,
                tuning_fold=tuning_fold,
                device=device,
            )
    except FloatingPointError as error:
        row = _failed_row(
            training_fold,
            parameter_count,
            model_seed,
            training_config,
            started,
            device,
            error,
        )
        del model
        _clear_cuda_cache(device)
        return row

    best = history.best_epoch - 1
    row = {
        "n_samples": len(training_fold.targets),
        "parameter_count": parameter_count,
        "random_seed": model_seed,
        "training_loss": float(history.training_loss[best]),
        "tune_loss": float(history.validation_loss[best]),
        "training_macro_f1": float(history.training_macro_f1[best]),
        "tune_macro_f1": float(history.validation_macro_f1[best]),
        "best_epoch": history.best_epoch,
        "epochs_run": len(history.training_loss),
        "batch_size": training_config.batch_size,
        "max_epochs": training_config.epochs,
        "patience": training_config.patience,
        "scheduler_strategy": training_config.scheduler_strategy,
        "training_loss_curve": dumps(history.training_loss),
        "tune_loss_curve": dumps(history.validation_loss),
        "training_accuracy_curve": dumps(history.training_accuracy),
        "tune_accuracy_curve": dumps(history.validation_accuracy),
        "duration_seconds": perf_counter() - started,
        "peak_memory_bytes": _peak_memory_bytes(device),
        "failed": False,
        "failure_reason": None,
    }
    del model
    _clear_cuda_cache(device)
    return row


def _failed_row(
    training_fold: FoldData,
    parameter_count: int,
    model_seed: int,
    training_config: TrainingConfig,
    started: float,
    device: torch.device,
    error: FloatingPointError,
) -> dict[str, object]:
    return {
        "n_samples": len(training_fold.targets),
        "parameter_count": parameter_count,
        "random_seed": model_seed,
        "training_loss": np.nan,
        "tune_loss": np.nan,
        "training_macro_f1": np.nan,
        "tune_macro_f1": np.nan,
        "best_epoch": 0,
        "epochs_run": 0,
        "batch_size": training_config.batch_size,
        "max_epochs": training_config.epochs,
        "patience": training_config.patience,
        "scheduler_strategy": training_config.scheduler_strategy,
        "training_loss_curve": "[]",
        "tune_loss_curve": "[]",
        "training_accuracy_curve": "[]",
        "tune_accuracy_curve": "[]",
        "duration_seconds": perf_counter() - started,
        "peak_memory_bytes": _peak_memory_bytes(device),
        "failed": True,
        "failure_reason": str(error),
    }


def _balanced_sample_indices(
    targets: Tensor,
    sample_count: int,
    *,
    random_seed: int,
) -> np.ndarray:
    values = _target_array(targets)
    if not 1 <= sample_count <= len(values):
        raise ValueError("sample_count must lie within the dataset size.")
    rng = np.random.default_rng(random_seed)
    classes = np.unique(values)
    pools = {
        int(label): rng.permutation(np.flatnonzero(values == label))
        for label in classes
    }
    positions = {int(label): 0 for label in classes}
    selected: list[int] = []
    while len(selected) < sample_count:
        for label in classes:
            key = int(label)
            if positions[key] >= len(pools[key]):
                continue
            selected.append(int(pools[key][positions[key]]))
            positions[key] += 1
            if len(selected) == sample_count:
                break
    return np.sort(np.asarray(selected, dtype=np.int64))


def _scaling_surface(
    inputs: tuple[np.ndarray, np.ndarray],
    irreducible_loss: float,
    data_coefficient: float,
    data_exponent: float,
    parameter_coefficient: float,
    parameter_exponent: float,
) -> np.ndarray:
    n_samples, parameters = inputs
    return (
        irreducible_loss
        + data_coefficient * n_samples ** (-data_exponent)
        + parameter_coefficient * parameters ** (-parameter_exponent)
    )


def _validate_fractions(fractions: Sequence[float]) -> None:
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("data fractions must be non-empty and lie in (0, 1].")
    if any(left >= right for left, right in zip(fractions, fractions[1:])):
        raise ValueError("data fractions must be strictly increasing.")


def _validate_positive_unique(values: Sequence[float], name: str) -> None:
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique positive values.")


def _target_array(targets: Tensor) -> np.ndarray:
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("targets must be a non-empty one-dimensional tensor.")
    return targets.detach().cpu().numpy().astype(np.int64, copy=False)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns {sorted(missing)}.")


def _peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _print_scaling_row(row: Mapping[str, object]) -> None:
    if row["failed"]:
        print(
            f"N={row['n_samples']:5d} w={row['width_multiplier']:g} "
            f"lr={row['learning_rate']:.1e}: failed"
        )
        return
    print(
        f"N={row['n_samples']:5d} P={row['parameter_count']:8d} "
        f"w={row['width_multiplier']:g} lr={row['learning_rate']:.1e} "
        f"tune_loss={row['tune_loss']:.4f}"
    )
