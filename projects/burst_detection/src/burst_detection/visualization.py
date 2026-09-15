"""Plots for trajectories, segmentation probabilities and event intervals."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .data import mask_to_events
from .metrics import probabilities_to_mask


def plot_burst_examples(
    signals: NDArray[np.floating],
    true_labels: NDArray[np.bool_] | NDArray[np.integer],
    *,
    probabilities: NDArray[np.floating] | None = None,
    threshold: float = 0.5,
    time_minutes: NDArray[np.floating] | None = None,
    maximum_examples: int = 6,
) -> tuple[Figure, NDArray[np.object_]]:
    """Overlay truth, probabilities and thresholded predicted events."""
    signal_values = np.asarray(signals)
    truth = np.asarray(true_labels, dtype=bool)
    if signal_values.shape != truth.shape or signal_values.ndim != 2:
        raise ValueError("signals and labels must share shape (samples, time).")
    if probabilities is not None and np.asarray(probabilities).shape != truth.shape:
        raise ValueError("probabilities must have the same shape as signals.")
    count = min(maximum_examples, len(signal_values))
    if count < 1:
        raise ValueError("at least one example is required.")
    times = (
        np.arange(signal_values.shape[1])
        if time_minutes is None
        else np.asarray(time_minutes)
    )
    if times.shape != (signal_values.shape[1],):
        raise ValueError("time_minutes must contain one value per time point.")
    predicted = (
        probabilities_to_mask(np.asarray(probabilities), threshold=threshold)
        if probabilities is not None
        else None
    )
    figure, axes = plt.subplots(count, 1, figsize=(11, 2.6 * count), squeeze=False)
    for index, axis in enumerate(axes[:, 0]):
        _shade_events(axis, times, truth[index], color="tab:green", alpha=0.18)
        if predicted is not None:
            _shade_events(axis, times, predicted[index], color="tab:red", alpha=0.14)
        axis.plot(times, signal_values[index], color="black", linewidth=1.3)
        axis.set_ylabel("SMAD ratio")
        axis.set_title(f"trajectory {index}: green=true, red=predicted")
        if probabilities is not None:
            probability_axis = axis.twinx()
            probability_axis.plot(
                times,
                np.asarray(probabilities)[index],
                color="tab:blue",
                linewidth=1.0,
                alpha=0.8,
            )
            probability_axis.axhline(threshold, color="tab:blue", linestyle=":")
            probability_axis.set_ylim(-0.02, 1.02)
            probability_axis.set_ylabel("P(burst)", color="tab:blue")
    axes[-1, 0].set_xlabel("time / min" if time_minutes is not None else "time point")
    figure.tight_layout()
    return figure, axes


def plot_training_history(
    history: object,
    *,
    title: str = "Training history",
) -> tuple[Figure, Axes]:
    """Plot train/tune losses from a ``TrainingHistory``-like object."""
    training_loss = np.asarray(getattr(history, "training_loss"))
    validation_loss = np.asarray(getattr(history, "validation_loss"))
    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(training_loss) + 1)
    axis.plot(epochs, training_loss, label="train")
    axis.plot(epochs, validation_loss, label="validation")
    best_epoch = getattr(history, "best_epoch", None)
    if best_epoch is not None:
        axis.axvline(
            best_epoch,
            color="black",
            linestyle=":",
            label="best validation epoch",
        )
    axis.set(xlabel="epoch", ylabel="weighted BCE", title=title)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_probability_comparison(
    signals: NDArray[np.floating],
    true_labels: NDArray[np.bool_] | NDArray[np.integer],
    probabilities: Mapping[str, NDArray[np.floating]],
    thresholds: Mapping[str, float],
    *,
    time_minutes: NDArray[np.floating] | None = None,
    maximum_examples: int = 4,
) -> tuple[Figure, NDArray[np.object_]]:
    """Compare model probabilities on the same hold-out trajectories."""
    signal_values = np.asarray(signals)
    truth = np.asarray(true_labels, dtype=bool)
    if signal_values.shape != truth.shape or signal_values.ndim != 2:
        raise ValueError("signals and labels must share shape (samples, time).")
    names = tuple(probabilities)
    if not names or set(names) != set(thresholds):
        raise ValueError("probabilities and thresholds must contain the same models.")
    for name, values in probabilities.items():
        if np.asarray(values).shape != truth.shape:
            raise ValueError(f"Probabilities for {name!r} have the wrong shape.")
    count = min(maximum_examples, len(signal_values))
    if count < 1:
        raise ValueError("at least one example is required.")
    times = (
        np.arange(signal_values.shape[1])
        if time_minutes is None
        else np.asarray(time_minutes)
    )
    figure, axes = plt.subplots(
        count,
        len(names),
        figsize=(min(10, 3.2 * len(names)), 2.4 * count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row in range(count):
        for column, name in enumerate(names):
            axis = axes[row, column]
            values = np.asarray(probabilities[name])[row]
            predicted = probabilities_to_mask(
                values[None, :], threshold=thresholds[name]
            )[0]
            _shade_events(axis, times, truth[row], color="tab:green", alpha=0.16)
            _shade_events(axis, times, predicted, color="tab:red", alpha=0.12)
            axis.plot(times, values, color="tab:blue", linewidth=1.1)
            axis.axhline(thresholds[name], color="tab:blue", linestyle=":")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.15)
            if row == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel(f"sample {row}\nP(burst)")
    for axis in axes[-1]:
        axis.set_xlabel("time / min" if time_minutes is not None else "time point")
    figure.suptitle("Hold-out probabilities: green=true, red=predicted")
    figure.tight_layout()
    return figure, axes


def _shade_events(
    axis: Axes,
    times: NDArray[np.floating],
    mask: NDArray[np.bool_],
    *,
    color: str,
    alpha: float,
) -> None:
    for event in mask_to_events(mask):
        left = times[event.start]
        right = times[min(event.stop, len(times) - 1)]
        if event.stop == len(times):
            right = times[-1]
        axis.axvspan(left, right, color=color, alpha=alpha, linewidth=0)
