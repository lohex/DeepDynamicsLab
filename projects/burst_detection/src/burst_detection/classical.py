"""Python equivalent of the classical ``stochastic-smad`` burst detector."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths


@dataclass(frozen=True, slots=True)
class ClassicalDetectorConfig:
    """Published defaults from ``+burstDetection/burstDetect.m``."""

    high_filter_width: float = 49.8453
    high_anchor_weight: int = 15
    low_filter_width: float = 10.1349
    low_anchor_weight: int = 20
    final_filter_width: float = 3.8217
    final_anchor_weight: int = 48
    low_peak_prominence: float = 0.0255
    low_peak_distance: float = 0.2776
    final_peak_prominence: float = 0.0655
    final_peak_distance: float = 11.0694
    label_width_factor: float = 0.7884


def detect_classical_bursts(
    trajectories: NDArray[np.floating],
    *,
    config: ClassicalDetectorConfig | None = None,
    timepoints: NDArray[np.floating] | None = None,
) -> NDArray[np.bool_]:
    """Detect bursts using the reference detector's smooth/detrend/peak pipeline.

    SciPy's half-prominence peak widths replace MATLAB ``findpeaks`` widths;
    Gaussian smoothing and the remaining equations are ported directly.
    """
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("trajectories must be a finite (samples, time) array.")
    cfg = config or ClassicalDetectorConfig()
    times = (
        np.arange(1, values.shape[1] + 1, dtype=np.float64)
        if timepoints is None
        else np.asarray(timepoints, dtype=np.float64)
    )
    if times.shape != (values.shape[1],) or len(times) < 2:
        raise ValueError("timepoints must contain one coordinate per sample.")
    steps = np.diff(times)
    if not np.allclose(steps, steps[0]) or steps[0] <= 0:
        raise ValueError("timepoints must be uniformly increasing.")
    time_step = float(steps[0])
    detected = np.zeros_like(values, dtype=bool)

    for sample_index, trajectory in enumerate(values):
        smooth = _gaussian_estimate(
            trajectory, times, cfg.low_filter_width, cfg.low_anchor_weight
        )
        low_peaks, low_properties = find_peaks(
            smooth,
            prominence=cfg.low_peak_prominence,
            distance=max(1, int(np.ceil(cfg.low_peak_distance / time_step))),
        )
        if len(low_peaks):
            low_widths = peak_widths(smooth, low_peaks, rel_height=0.5)[0] * time_step
            low_prominences = low_properties["prominences"]
            hills = np.stack(
                [
                    prominence
                    * np.exp(-((times - times[peak]) / width) ** 2)
                    for peak, width, prominence in zip(
                        low_peaks, low_widths, low_prominences
                    )
                ]
            ).sum(axis=0)
        else:
            hills = np.zeros_like(trajectory)
        trend = _gaussian_estimate(
            smooth - hills, times, cfg.high_filter_width, cfg.high_anchor_weight
        )
        detrended = trajectory - trend
        detrended -= detrended.min()
        final_signal = _gaussian_estimate(
            detrended, times, cfg.final_filter_width, cfg.final_anchor_weight
        )
        peaks, _ = find_peaks(
            final_signal,
            prominence=cfg.final_peak_prominence,
            distance=max(1, int(np.ceil(cfg.final_peak_distance / time_step))),
        )
        if not len(peaks):
            continue
        widths = peak_widths(final_signal, peaks, rel_height=0.5)[0] * time_step
        for peak, width in zip(peaks, widths):
            support = np.abs(times - times[peak]) <= cfg.label_width_factor * width
            detected[sample_index, support] = True
    return detected


def _gaussian_estimate(
    signal: NDArray[np.floating],
    timepoints: NDArray[np.floating],
    width: float,
    anchor_weight: int,
) -> NDArray[np.float64]:
    """Port ``gauss_est`` including repeated endpoint anchors."""
    if width <= 0 or anchor_weight < 0:
        raise ValueError("width must be positive and anchor_weight non-negative.")
    weights = _smoothing_weights(
        len(signal),
        float(timepoints[0]),
        float(timepoints[1] - timepoints[0]),
        float(width),
        int(anchor_weight),
    )
    extended = np.pad(np.asarray(signal, dtype=np.float64), anchor_weight, mode="edge")
    return weights @ extended


@lru_cache(maxsize=32)
def _smoothing_weights(
    length: int,
    first_time: float,
    time_step: float,
    width: float,
    anchor_weight: int,
) -> NDArray[np.float64]:
    output_times = first_time + np.arange(length, dtype=np.float64) * time_step
    extended_times = first_time + (
        np.arange(-anchor_weight, length + anchor_weight, dtype=np.float64)
        * time_step
    )
    distances = (output_times[:, None] - extended_times[None, :]) / width
    weights = np.exp(-0.5 * distances**2)
    return weights / weights.sum(axis=1, keepdims=True)
