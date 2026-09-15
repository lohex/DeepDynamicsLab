"""Reproducible training and checkpointing for temporal burst segmentation."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TrajectoryPreprocessor
from .model import BurstTCN, create_burst_model


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    positive_weight: float | Literal["auto"] = "auto"
    maximum_positive_weight: float = 20.0
    gradient_clip_norm: float | None = 1.0
    random_seed: int = 42
    report_every: int = 5


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    validation_average_precision: tuple[float, ...]
    best_epoch: int
    positive_weight: float


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics exposed after each epoch for monitoring and trial pruning."""

    epoch: int
    training_loss: float
    validation_loss: float
    validation_average_precision: float


EpochCallback = Callable[[EpochMetrics], None]


@dataclass(frozen=True, slots=True)
class SanityCheckResult:
    initial_loss: float
    final_loss: float
    pointwise_f1: float
    passed: bool


def artifact_root() -> Path:
    """Return this project's notebook artifact directory."""
    return Path(__file__).resolve().parents[2] / "notebooks" / "artifacts"


def set_reproducible_seed(random_seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic CUDA kernels."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(
    model: nn.Module,
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    config: TrainingConfig | None = None,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = None,
    preprocessor: TrajectoryPreprocessor | None = None,
    epoch_callback: EpochCallback | None = None,
) -> TrainingHistory:
    """Train with weighted BCE, early stopping and best-weight restoration.

    The function seeds minibatch order and stochastic layers. Seed model
    construction with :func:`set_reproducible_seed` when reproducible initial
    weights are required.
    """
    cfg = config or TrainingConfig()
    _validate_inputs(
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        cfg,
    )
    set_reproducible_seed(cfg.random_seed)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    generator = torch.Generator().manual_seed(cfg.random_seed)
    training_loader = DataLoader(
        TensorDataset(training_features.float(), training_targets.float()),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=resolved_device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features.float(), validation_targets.float()),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=resolved_device.type == "cuda",
    )

    positive_weight = _resolve_positive_weight(training_targets, cfg)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=resolved_device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    model.to(resolved_device)

    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_average_precisions: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        training_loss, _ = _run_epoch(
            model,
            training_loader,
            loss_function,
            resolved_device,
            optimizer=optimizer,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )
        validation_loss, validation_probabilities = _run_epoch(
            model, validation_loader, loss_function, resolved_device
        )
        validation_ap = float(
            average_precision_score(
                validation_targets.detach().cpu().numpy().reshape(-1),
                validation_probabilities.reshape(-1),
            )
        )
        if not np.isfinite((training_loss, validation_loss, validation_ap)).all():
            raise FloatingPointError(f"Non-finite metric at epoch {epoch}.")
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        validation_average_precisions.append(validation_ap)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = deepcopy(model.state_dict())
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    training_config=cfg,
                    positive_weight=positive_weight,
                    preprocessor=preprocessor,
                )
        else:
            stale_epochs += 1

        if epoch_callback is not None:
            epoch_callback(
                EpochMetrics(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                    validation_average_precision=validation_ap,
                )
            )

        if epoch == 1 or epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:3d}: train loss {training_loss:.4f}, "
                f"validation loss {validation_loss:.4f}, AP {validation_ap:.3f}"
            )
        if stale_epochs >= cfg.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return TrainingHistory(
        training_loss=tuple(training_losses),
        validation_loss=tuple(validation_losses),
        validation_average_precision=tuple(validation_average_precisions),
        best_epoch=best_epoch,
        positive_weight=positive_weight,
    )


def predict_probabilities(
    model: nn.Module,
    features: Tensor,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Return pointwise sigmoid probabilities in bounded batches."""
    if features.ndim != 3 or len(features) < 1 or batch_size < 1:
        raise ValueError(
            "features must be a non-empty three-dimensional tensor and "
            "batch_size positive."
        )
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = DataLoader(TensorDataset(features.float()), batch_size=batch_size)
    model.to(resolved_device)
    model.eval()
    batches: list[Tensor] = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(resolved_device, non_blocking=True))
            batches.append(torch.sigmoid(logits).cpu())
    return torch.cat(batches).numpy()


def overfit_sanity_check(
    features: Tensor,
    targets: Tensor,
    *,
    sample_count: int = 8,
    epochs: int = 120,
    random_seed: int = 7,
    device: torch.device | str = "cpu",
) -> SanityCheckResult:
    """Try to memorize a tiny fixed batch before committing to a full run."""
    if not 1 <= sample_count <= len(features):
        raise ValueError("sample_count must fit inside the supplied data.")
    set_reproducible_seed(random_seed)
    tiny_features = features[:sample_count].float()
    tiny_targets = targets[:sample_count].float()
    model = BurstTCN(
        input_channels=tiny_features.shape[1],
        hidden_channels=16,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
    ).to(device)
    positive_weight = max(
        1.0,
        float((tiny_targets.numel() - tiny_targets.sum()) / tiny_targets.sum().clamp_min(1)),
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch_features = tiny_features.to(device)
    batch_targets = tiny_targets.to(device)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = loss_function(logits, batch_targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        predictions = (torch.sigmoid(model(batch_features)) >= 0.5).cpu().numpy()
    score = float(f1_score(tiny_targets.numpy().reshape(-1), predictions.reshape(-1)))
    passed = losses[-1] < 0.5 * losses[0] and score >= 0.90
    return SanityCheckResult(losses[0], losses[-1], score, passed)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    validation_loss: float,
    training_config: TrainingConfig,
    positive_weight: float,
    preprocessor: TrajectoryPreprocessor | None = None,
) -> Path:
    """Save enough model metadata to reconstruct the best TCN."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": getattr(model, "model_name", type(model).__name__),
            "model_config": getattr(model, "model_config", None),
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "epoch": epoch,
            "validation_loss": validation_loss,
            "positive_weight": positive_weight,
            "training_config": asdict(training_config),
            "preprocessor": asdict(preprocessor) if preprocessor is not None else None,
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, object]]:
    """Rebuild a supported burst-segmentation model from its checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_name, str) or not isinstance(model_config, dict):
        raise ValueError("checkpoint has no supported model metadata.")
    model = create_burst_model(model_name, model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, np.ndarray]:
    learning = optimizer is not None
    model.train(learning)
    total_loss = 0.0
    sample_count = 0
    probabilities: list[Tensor] = []
    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        if learning:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(learning):
            logits = model(batch_features)
            loss = loss_function(logits, batch_targets)
        if learning:
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        total_loss += float(loss.detach()) * len(batch_targets)
        sample_count += len(batch_targets)
        probabilities.append(torch.sigmoid(logits.detach()).cpu())
    return total_loss / sample_count, torch.cat(probabilities).numpy()


def _resolve_positive_weight(targets: Tensor, config: TrainingConfig) -> float:
    if config.positive_weight == "auto":
        positives = float(targets.sum())
        negatives = float(targets.numel() - positives)
        if positives <= 0:
            raise ValueError("automatic positive weighting requires positive labels.")
        return min(config.maximum_positive_weight, max(1.0, negatives / positives))
    return float(config.positive_weight)


def _validate_inputs(
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    config: TrainingConfig,
) -> None:
    for features, targets, name in (
        (training_features, training_targets, "training"),
        (validation_features, validation_targets, "validation"),
    ):
        if features.ndim != 3 or targets.ndim != 2:
            raise ValueError(f"{name} features/targets must be 3D/2D.")
        if len(features) != len(targets) or features.shape[-1] != targets.shape[-1]:
            raise ValueError(f"{name} features and targets are not aligned.")
    if training_features.shape[1:] != validation_features.shape[1:]:
        raise ValueError("training and validation feature shapes must match.")
    if min(config.epochs, config.batch_size, config.patience, config.report_every) < 1:
        raise ValueError("epochs, batch size, patience and report interval must be positive.")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    if config.positive_weight != "auto" and float(config.positive_weight) <= 0:
        raise ValueError("positive_weight must be positive.")
    if config.maximum_positive_weight < 1:
        raise ValueError("maximum_positive_weight must be at least one.")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None.")
