"""Pointwise and event-level metrics for burst segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .data import BurstEvent, mask_to_events


@dataclass(frozen=True, slots=True)
class BurstMetrics:
    point_precision: float
    point_recall: float
    point_f1: float
    average_precision: float
    roc_auc: float
    event_precision: float
    event_recall: float
    event_f1: float
    true_event_count: int
    predicted_event_count: int
    matched_event_count: int
    burst_count_mae: float
    center_time_mae_minutes: float
    duration_mae_minutes: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    metrics: BurstMetrics


def probabilities_to_mask(
    probabilities: NDArray[np.floating],
    *,
    threshold: float = 0.5,
    minimum_duration: int = 3,
    bridge_gap: int = 1,
) -> NDArray[np.bool_]:
    """Threshold, close short internal gaps and reject very short events."""
    values = np.asarray(probabilities)
    if values.ndim != 2:
        raise ValueError("probabilities must have shape (samples, time).")
    if not 0 <= threshold <= 1 or minimum_duration < 1 or bridge_gap < 0:
        raise ValueError("invalid threshold or event post-processing settings.")
    masks = values >= threshold
    cleaned = np.zeros_like(masks, dtype=bool)
    for sample_index, row in enumerate(masks):
        closed = row.copy()
        if bridge_gap:
            events = mask_to_events(closed)
            for left, right in zip(events, events[1:]):
                if right.start - left.stop <= bridge_gap:
                    closed[left.stop : right.start] = True
        for event in mask_to_events(closed):
            if event.duration >= minimum_duration:
                cleaned[sample_index, event.start : event.stop] = True
    return cleaned


def evaluate_burst_predictions(
    true_labels: NDArray[np.bool_] | NDArray[np.integer],
    probabilities: NDArray[np.floating],
    *,
    threshold: float = 0.5,
    minimum_duration: int = 3,
    bridge_gap: int = 1,
    minimum_iou: float = 0.1,
    sample_interval_minutes: float = 5.0,
) -> BurstMetrics:
    """Compute point metrics and one-to-one interval matching metrics."""
    truth = np.asarray(true_labels, dtype=bool)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != scores.shape or truth.ndim != 2:
        raise ValueError("true_labels and probabilities must share a 2D shape.")
    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must lie in [0, 1].")
    predicted = probabilities_to_mask(
        scores,
        threshold=threshold,
        minimum_duration=minimum_duration,
        bridge_gap=bridge_gap,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth.reshape(-1),
        predicted.reshape(-1),
        average="binary",
        zero_division=0,
    )
    flat_truth = truth.reshape(-1)
    flat_scores = scores.reshape(-1)
    average_precision = float(average_precision_score(flat_truth, flat_scores))
    roc_auc = (
        float(roc_auc_score(flat_truth, flat_scores))
        if np.unique(flat_truth).size == 2
        else float("nan")
    )

    true_count = 0
    predicted_count = 0
    matched_count = 0
    count_errors: list[int] = []
    center_errors: list[float] = []
    duration_errors: list[float] = []
    for true_row, predicted_row in zip(truth, predicted):
        true_events = mask_to_events(true_row)
        predicted_events = mask_to_events(predicted_row)
        matches = match_events(true_events, predicted_events, minimum_iou=minimum_iou)
        true_count += len(true_events)
        predicted_count += len(predicted_events)
        matched_count += len(matches)
        count_errors.append(abs(len(predicted_events) - len(true_events)))
        for true_index, predicted_index in matches:
            true_event = true_events[true_index]
            predicted_event = predicted_events[predicted_index]
            center_errors.append(abs(predicted_event.center - true_event.center))
            duration_errors.append(abs(predicted_event.duration - true_event.duration))

    event_precision = matched_count / predicted_count if predicted_count else 0.0
    event_recall = matched_count / true_count if true_count else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    return BurstMetrics(
        point_precision=float(precision),
        point_recall=float(recall),
        point_f1=float(f1),
        average_precision=average_precision,
        roc_auc=roc_auc,
        event_precision=event_precision,
        event_recall=event_recall,
        event_f1=event_f1,
        true_event_count=true_count,
        predicted_event_count=predicted_count,
        matched_event_count=matched_count,
        burst_count_mae=float(np.mean(count_errors)),
        center_time_mae_minutes=_mean_or_nan(center_errors) * sample_interval_minutes,
        duration_mae_minutes=_mean_or_nan(duration_errors) * sample_interval_minutes,
    )


def match_events(
    true_events: list[BurstEvent],
    predicted_events: list[BurstEvent],
    *,
    minimum_iou: float = 0.1,
) -> list[tuple[int, int]]:
    """Match intervals one-to-one by maximum total intersection-over-union."""
    if not true_events or not predicted_events:
        return []
    ious = np.asarray(
        [
            [_interval_iou(true_event, predicted_event) for predicted_event in predicted_events]
            for true_event in true_events
        ],
        dtype=np.float64,
    )
    true_indices, predicted_indices = linear_sum_assignment(-ious)
    return [
        (int(true_index), int(predicted_index))
        for true_index, predicted_index in zip(true_indices, predicted_indices)
        if ious[true_index, predicted_index] >= minimum_iou
    ]


def select_threshold(
    true_labels: NDArray[np.bool_] | NDArray[np.integer],
    probabilities: NDArray[np.floating],
    *,
    thresholds: NDArray[np.floating] | None = None,
    minimum_duration: int = 3,
    bridge_gap: int = 1,
    minimum_iou: float = 0.1,
    sample_interval_minutes: float = 5.0,
) -> ThresholdSelection:
    """Choose the event-F1 threshold on validation data only."""
    candidates = (
        np.asarray(thresholds, dtype=np.float64)
        if thresholds is not None
        else np.linspace(0.1, 0.9, 17)
    )
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array.")
    selections = [
        ThresholdSelection(
            float(threshold),
            evaluate_burst_predictions(
                true_labels,
                probabilities,
                threshold=float(threshold),
                minimum_duration=minimum_duration,
                bridge_gap=bridge_gap,
                minimum_iou=minimum_iou,
                sample_interval_minutes=sample_interval_minutes,
            ),
        )
        for threshold in candidates
    ]
    return max(
        selections,
        key=lambda selection: (
            selection.metrics.event_f1,
            selection.metrics.point_f1,
            -abs(selection.threshold - 0.5),
        ),
    )


def _interval_iou(left: BurstEvent, right: BurstEvent) -> float:
    intersection = max(0, min(left.stop, right.stop) - max(left.start, right.start))
    union = left.duration + right.duration - intersection
    return intersection / union if union else 0.0


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
