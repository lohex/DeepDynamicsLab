"""Manifests connecting validation-selected models to hold-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .data import TrajectoryPreprocessor


def save_holdout_manifest(
    path: str | Path,
    *,
    candidates: Sequence[Mapping[str, object]],
    split_config: Mapping[str, int],
) -> Path:
    """Persist frozen checkpoints and validation-selected operating points."""
    if not candidates:
        raise ValueError("At least one model candidate is required.")
    required = {"architecture", "checkpoint", "threshold", "validation_metrics"}
    for candidate in candidates:
        missing = required.difference(candidate)
        if missing:
            raise ValueError(f"Candidate is missing fields: {sorted(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "selection_fold": "validation",
        "threshold_metric": "event_f1",
        "split_config": dict(split_config),
        "candidates": [dict(candidate) for candidate in candidates],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_holdout_manifest(path: str | Path) -> dict[str, object]:
    """Load and minimally validate a frozen hold-out manifest."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported burst hold-out manifest schema.")
    if payload.get("selection_fold") != "validation":
        raise ValueError("Burst candidates must be selected on validation.")
    if not payload.get("candidates"):
        raise ValueError("Burst hold-out manifest contains no candidates.")
    return payload


def preprocessor_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> TrajectoryPreprocessor:
    """Reconstruct train-fitted preprocessing saved with a model checkpoint."""
    values = checkpoint.get("preprocessor")
    if not isinstance(values, dict):
        raise ValueError("Checkpoint contains no serialized preprocessor.")
    return TrajectoryPreprocessor(**values)
