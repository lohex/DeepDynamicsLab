"""Optional MLflow tracking for Optuna-driven time-course experiments."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .training import EpochMetrics


@dataclass(frozen=True, slots=True)
class MLflowTrackingConfig:
    """Local MLflow destination and naming for one Optuna study."""

    tracking_uri: str
    experiment_name: str
    parent_run_name: str
    artifact_location: str | None = None

    @classmethod
    def local(
        cls,
        root: str | Path,
        *,
        experiment_name: str,
        parent_run_name: str,
    ) -> "MLflowTrackingConfig":
        """Build a portable local SQLite configuration below ``root``."""
        destination = Path(root).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        return cls(
            tracking_uri=f"sqlite:///{destination / 'mlflow.db'}",
            experiment_name=experiment_name,
            parent_run_name=parent_run_name,
            artifact_location=(destination / "mlruns").as_uri(),
        )


class MLflowTracker:
    """Small explicit logger; importing the project does not require MLflow."""

    def __init__(self, config: MLflowTrackingConfig) -> None:
        self.config = config
        self._mlflow = _require_mlflow()

    @contextmanager
    def study_run(self, *, scope: str, study_name: str) -> Iterator[None]:
        """Open the parent run that groups all nested Optuna trials."""
        mlflow = self._mlflow
        mlflow.set_tracking_uri(self.config.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.config.experiment_name,
                artifact_location=self.config.artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.config.parent_run_name,
            tags={"run_type": "optuna_study", "scan_scope": scope},
        ):
            mlflow.log_params({"study_name": study_name, "scan_scope": scope})
            yield

    @contextmanager
    def trial_run(
        self,
        *,
        trial_number: int,
        model_config: Mapping[str, object],
        training_config: object,
        parameter_count: int,
    ) -> Iterator[None]:
        """Open a nested run and record a complete proposed configuration."""
        mlflow = self._mlflow
        with mlflow.start_run(
            run_name=f"trial-{trial_number:04d}",
            nested=True,
            tags={"run_type": "optuna_trial", "trial_number": trial_number},
        ):
            flattened: dict[str, object] = {
                "trial_number": trial_number,
                "parameter_count": parameter_count,
            }
            flattened.update(_flatten("model", model_config))
            training_values = (
                asdict(training_config)
                if is_dataclass(training_config)
                else dict(training_config)  # type: ignore[arg-type]
            )
            flattened.update(_flatten("training", training_values))
            mlflow.log_params({key: _parameter_value(value) for key, value in flattened.items()})
            try:
                yield
            except BaseException:
                mlflow.set_tag("trial_status", "failed_or_pruned")
                raise
            else:
                mlflow.set_tag("trial_status", "complete")

    def log_epoch(self, metrics: EpochMetrics) -> None:
        """Log the shared epoch metrics with an explicit MLflow step."""
        self._mlflow.log_metrics(
            {
                "train_loss": metrics.training_loss,
                "tune_loss": metrics.validation_loss,
                "train_accuracy": metrics.training_accuracy,
                "tune_accuracy": metrics.validation_accuracy,
                "train_macro_f1": metrics.training_macro_f1,
                "tune_macro_f1": metrics.validation_macro_f1,
                "learning_rate": metrics.learning_rate,
            },
            step=metrics.epoch,
        )

    def log_trial_summary(self, metrics: Mapping[str, float | int]) -> None:
        """Log restored-checkpoint metrics after one successful trial."""
        self._mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
                if np.isfinite(float(value))
            }
        )


def _flatten(prefix: str, values: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}.{key}": value for key, value in values.items()}


def _parameter_value(value: object) -> object:
    if isinstance(value, (tuple, list, dict)):
        return repr(value)
    if value is None:
        return "none"
    return value


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MLflow is required for experiment tracking. Install with "
            "`uv sync --extra torch --extra experiment`."
        ) from error
    return mlflow
