"""Synthetic SMAD-like trajectories and train-only preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BurstEvent:
    """A half-open burst interval ``[start, stop)`` in sample coordinates."""

    start: int
    stop: int

    @property
    def center(self) -> float:
        return (self.start + self.stop - 1) / 2

    @property
    def duration(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class ArchetypeBank:
    """Base trajectories and their empirical sampling probabilities."""

    trajectories: FloatArray
    probabilities: FloatArray

    def __post_init__(self) -> None:
        trajectories = np.asarray(self.trajectories)
        probabilities = np.asarray(self.probabilities)
        if trajectories.ndim != 2 or len(trajectories) == 0:
            raise ValueError("trajectories must have shape (archetypes, time).")
        if probabilities.shape != (len(trajectories),):
            raise ValueError("probabilities must contain one value per archetype.")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one.")


@dataclass(frozen=True, slots=True)
class SyntheticBurstConfig:
    """Defaults translated from ``stochastic-smad/generateSample.m``.

    Positions are kept in MATLAB's one-based coordinates while generated arrays
    use normal zero-based indexing. ``height`` is the area multiplier of a
    normalized Gaussian density, matching MATLAB's ``normpdf * height``.
    """

    length: int = 289
    sample_interval_minutes: float = 5.0
    separation_factor: float = 4.5
    noise_scale: float = 0.015
    maximum_peak_attempts: int = 16
    center_min: int = 55
    center_max: int = 260
    width_min: float = 4.0
    width_range: float = 6.0
    height_min: float = 3.0
    height_range: float = 2.0
    initial_label_start: int = 7
    initial_label_stop: int = 40

    def __post_init__(self) -> None:
        if self.length < 3:
            raise ValueError("length must be at least three.")
        if not 1 <= self.center_min <= self.center_max <= self.length:
            raise ValueError("burst centers must lie inside the trajectory.")
        if not 1 <= self.initial_label_start <= self.initial_label_stop <= self.length:
            raise ValueError("the initial response interval must lie inside the trajectory.")
        if self.maximum_peak_attempts < 1:
            raise ValueError("maximum_peak_attempts must be positive.")
        if min(self.noise_scale, self.width_min, self.height_min) < 0:
            raise ValueError("noise, width and height parameters must be non-negative.")
        if self.width_min == 0:
            raise ValueError("width_min must be positive.")


@dataclass(frozen=True, slots=True)
class SyntheticBurstDataset:
    """Generated signals, binary masks, instance labels and event intervals."""

    signals: FloatArray
    labels: BoolArray
    instance_labels: IntArray
    events: tuple[tuple[BurstEvent, ...], ...]
    time_minutes: FloatArray


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: SyntheticBurstDataset
    validation: SyntheticBurstDataset
    test: SyntheticBurstDataset


@dataclass(frozen=True, slots=True)
class TrajectoryPreprocessor:
    """Channel-wise normalization fitted using training trajectories only."""

    signal_mean: float
    signal_std: float
    derivative_mean: float
    derivative_std: float
    include_derivative: bool = True

    def transform(self, signals: FloatArray) -> torch.Tensor:
        values = _validate_signals(signals)
        raw = (values - self.signal_mean) / self.signal_std
        channels = [raw]
        if self.include_derivative:
            derivative = np.diff(values, axis=1, prepend=values[:, :1])
            derivative = (derivative - self.derivative_mean) / self.derivative_std
            channels.append(derivative)
        features = np.stack(channels, axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(features)


def fit_archetype_bank(
    trajectories: FloatArray,
    *,
    n_archetypes: int = 3,
    random_seed: int = 42,
) -> ArchetypeBank:
    """Fit the three weighted K-means archetypes used by the MATLAB generator."""
    values = _validate_signals(trajectories)
    if n_archetypes < 1 or n_archetypes > len(values):
        raise ValueError("n_archetypes must be between one and the sample count.")
    kmeans = KMeans(n_clusters=n_archetypes, n_init=10, random_state=random_seed)
    assignments = kmeans.fit_predict(values)
    counts = np.bincount(assignments, minlength=n_archetypes)
    centers = kmeans.cluster_centers_
    # Stable ordering makes plots and serialized examples easier to compare.
    order = np.argsort(centers[:, -max(1, centers.shape[1] // 3) :].mean(axis=1))
    return ArchetypeBank(
        trajectories=np.asarray(centers[order], dtype=np.float32),
        probabilities=np.asarray(counts[order] / counts.sum(), dtype=np.float64),
    )


def default_smad_archetypes(length: int = 289) -> ArchetypeBank:
    """Return an analytic fallback bank with transient SMAD response shapes."""
    if length < 3:
        raise ValueError("length must be at least three.")
    time = np.arange(length, dtype=np.float64)
    response = (1.0 - np.exp(-time / 3.5)) * np.exp(-time / 28.0)
    slow_tail = 1.0 - np.exp(-time / 80.0)
    centers = np.stack(
        (
            0.36 + 1.35 * response + 0.20 * slow_tail,
            0.43 + 1.05 * response + 0.34 * slow_tail,
            0.32 + 1.65 * response + 0.12 * slow_tail,
        )
    )
    return ArchetypeBank(
        trajectories=centers.astype(np.float32),
        probabilities=np.asarray((0.34, 0.41, 0.25), dtype=np.float64),
    )


def generate_synthetic_bursts(
    sample_count: int,
    *,
    archetypes: ArchetypeBank | None = None,
    config: SyntheticBurstConfig | None = None,
    random_seed: int = 42,
) -> SyntheticBurstDataset:
    """Generate trajectories by porting the logic of MATLAB ``generateSample``."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    cfg = config or SyntheticBurstConfig()
    bank = archetypes or default_smad_archetypes(cfg.length)
    if bank.trajectories.shape[1] != cfg.length:
        raise ValueError("archetypes and config must have the same trajectory length.")

    rng = np.random.default_rng(random_seed)
    matlab_time = np.arange(1, cfg.length + 1, dtype=np.float64)
    signals = np.empty((sample_count, cfg.length), dtype=np.float32)
    instance_labels = np.zeros((sample_count, cfg.length), dtype=np.int64)

    for sample_index in range(sample_count):
        archetype_index = rng.choice(len(bank.trajectories), p=bank.probabilities)
        signal = bank.trajectories[archetype_index].astype(np.float64, copy=True)
        signal += np.cumsum(rng.normal(0.0, cfg.noise_scale, cfg.length))

        accepted: list[tuple[int, float]] = []
        attempt_count = int(rng.integers(1, cfg.maximum_peak_attempts + 1))
        for _ in range(attempt_count):
            center = int(rng.integers(cfg.center_min, cfg.center_max + 1))
            width = float(cfg.width_min + cfg.width_range * rng.random())
            height = float(cfg.height_min + cfg.height_range * rng.random())
            separations = [
                abs(center - old_center) / np.sqrt(width * old_width)
                for old_center, old_width in accepted
            ]
            if separations and min(separations) <= cfg.separation_factor:
                continue

            gaussian_density = np.exp(-0.5 * ((matlab_time - center) / width) ** 2)
            gaussian_density /= np.sqrt(2.0 * np.pi) * width
            signal += height * gaussian_density
            accepted.append((center, width))

            start_matlab = max(1, int(np.ceil(center - 2.0 * width)))
            stop_matlab = min(cfg.length, int(np.floor(center + 2.0 * width)))
            instance_labels[
                sample_index, start_matlab - 1 : stop_matlab
            ] = len(accepted) + 1

        # The original code assigns the stimulus-driven initial peak last.
        instance_labels[
            sample_index,
            cfg.initial_label_start - 1 : cfg.initial_label_stop,
        ] = 1
        signals[sample_index] = signal.astype(np.float32)

    labels = instance_labels > 0
    events = tuple(tuple(mask_to_events(mask)) for mask in labels)
    time_minutes = (
        np.arange(cfg.length, dtype=np.float32) * cfg.sample_interval_minutes
    )
    return SyntheticBurstDataset(signals, labels, instance_labels, events, time_minutes)


def generate_dataset_splits(
    *,
    train_size: int = 800,
    validation_size: int = 200,
    test_size: int = 200,
    archetypes: ArchetypeBank | None = None,
    config: SyntheticBurstConfig | None = None,
    random_seed: int = 42,
) -> DatasetSplits:
    """Generate independent, reproducible train/validation/test samples."""
    if min(train_size, validation_size, test_size) < 1:
        raise ValueError("all split sizes must be positive.")
    child_seeds = np.random.SeedSequence(random_seed).spawn(3)

    def make(size: int, seed: np.random.SeedSequence) -> SyntheticBurstDataset:
        resolved_seed = int(seed.generate_state(1, dtype=np.uint32)[0])
        return generate_synthetic_bursts(
            size,
            archetypes=archetypes,
            config=config,
            random_seed=resolved_seed,
        )

    return DatasetSplits(
        train=make(train_size, child_seeds[0]),
        validation=make(validation_size, child_seeds[1]),
        test=make(test_size, child_seeds[2]),
    )


def fit_preprocessor(
    training_signals: FloatArray,
    *,
    include_derivative: bool = True,
) -> TrajectoryPreprocessor:
    """Estimate normalization statistics exclusively from the training split."""
    values = _validate_signals(training_signals).astype(np.float64, copy=False)
    derivative = np.diff(values, axis=1, prepend=values[:, :1])
    signal_std = float(values.std())
    derivative_std = float(derivative.std())
    if signal_std <= 0 or derivative_std <= 0:
        raise ValueError("training signals and derivatives must have positive variance.")
    return TrajectoryPreprocessor(
        signal_mean=float(values.mean()),
        signal_std=signal_std,
        derivative_mean=float(derivative.mean()),
        derivative_std=derivative_std,
        include_derivative=include_derivative,
    )


def targets_to_tensor(labels: BoolArray) -> torch.Tensor:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("labels must have shape (samples, time).")
    return torch.from_numpy(values.astype(np.float32, copy=False))


def mask_to_events(mask: NDArray[np.bool_] | NDArray[np.integer]) -> list[BurstEvent]:
    """Convert a one-dimensional positive mask to contiguous half-open events."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [BurstEvent(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _validate_signals(signals: FloatArray) -> FloatArray:
    values = np.asarray(signals)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("signals must have shape (samples, time).")
    if not np.isfinite(values).all():
        raise ValueError("signals must be finite.")
    return values
