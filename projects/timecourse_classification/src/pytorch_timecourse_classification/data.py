"""Shared preparation of fixed time-course classification folds."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from deeplearning_examples.io import FloatArray, Fold, LabelArray, load_data


IntArray = NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class FoldData:
    """Raw trajectories, metadata and model-ready tensors for one fold."""

    trajectories: FloatArray
    doses: LabelArray
    years: IntArray
    features: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class Preprocessor:
    """Training-derived normalization and dose encoding."""

    class_names: tuple[str, ...]
    signal_mean: float
    signal_std: float

    def transform_features(self, trajectories: FloatArray) -> Tensor:
        normalized = (trajectories - self.signal_mean) / self.signal_std
        return torch.from_numpy(normalized[:, np.newaxis, :]).float()

    def transform_targets(self, doses: LabelArray) -> Tensor:
        class_to_index = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        try:
            encoded = [class_to_index[str(dose)] for dose in doses]
        except KeyError as error:
            raise ValueError(f"Unknown dose label {error.args[0]!r}.") from error
        return torch.tensor(encoded, dtype=torch.long)


def preprocessors_compatible(
    reference: Preprocessor,
    candidate: Preprocessor,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """Check semantic compatibility without requiring bit-identical floats."""
    return (
        reference.class_names == candidate.class_names
        and bool(
            np.isclose(
                reference.signal_mean,
                candidate.signal_mean,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
        and bool(
            np.isclose(
                reference.signal_std,
                candidate.signal_std,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
    )


def load_training_folds() -> tuple[FoldData, FoldData, Preprocessor]:
    """Load train and validate folds and fit preprocessing on train only."""
    train_trajectories, train_doses, train_years = load_data("train")
    tune_trajectories, tune_doses, tune_years = load_data("validate")
    preprocessor = fit_preprocessor(train_trajectories, train_doses)
    return (
        _prepare_fold(train_trajectories, train_doses, train_years, preprocessor),
        _prepare_fold(tune_trajectories, tune_doses, tune_years, preprocessor),
        preprocessor,
    )


def load_prepared_fold(fold: Fold, preprocessor: Preprocessor) -> FoldData:
    """Load one fixed fold using training-derived preprocessing."""
    trajectories, doses, years = load_data(fold)
    return _prepare_fold(trajectories, doses, years, preprocessor)


def fit_preprocessor(
    training_trajectories: FloatArray,
    training_doses: LabelArray,
) -> Preprocessor:
    """Fit global normalization and a numeric dose ordering on train only."""
    class_names = tuple(
        sorted(
            (str(value) for value in np.unique(training_doses)),
            key=lambda value: float(value.removesuffix("pM")),
        )
    )
    signal_mean = float(training_trajectories.mean())
    signal_std = float(training_trajectories.std())
    if not np.isfinite(signal_mean) or not np.isfinite(signal_std) or signal_std <= 0:
        raise ValueError(
            "Training trajectories must have a finite, positive standard deviation."
        )
    return Preprocessor(class_names, signal_mean, signal_std)


def subset_fold(fold: FoldData, indices: Sequence[int] | IntArray) -> FoldData:
    """Select the same sample indices from all raw and model-ready fold fields."""
    resolved = np.asarray(indices, dtype=np.int64)
    if resolved.ndim != 1 or len(resolved) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence.")
    if len(np.unique(resolved)) != len(resolved):
        raise ValueError("indices must be unique.")
    if resolved.min() < 0 or resolved.max() >= len(fold.targets):
        raise IndexError("indices contain values outside the fold.")
    feature_indices = torch.as_tensor(
        resolved, dtype=torch.long, device=fold.features.device
    )
    target_indices = torch.as_tensor(
        resolved, dtype=torch.long, device=fold.targets.device
    )
    return FoldData(
        trajectories=fold.trajectories[resolved],
        doses=fold.doses[resolved],
        years=fold.years[resolved],
        features=fold.features.index_select(0, feature_indices),
        targets=fold.targets.index_select(0, target_indices),
    )


def _prepare_fold(
    trajectories: FloatArray,
    doses: LabelArray,
    years: IntArray,
    preprocessor: Preprocessor,
) -> FoldData:
    return FoldData(
        trajectories=trajectories,
        doses=doses,
        years=years,
        features=preprocessor.transform_features(trajectories),
        targets=preprocessor.transform_targets(doses),
    )
