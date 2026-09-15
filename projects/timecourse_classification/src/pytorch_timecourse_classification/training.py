"""Function-based training for the PyTorch time-course classifier."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Literal

import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    scheduler_step_size: int = 10
    learning_rate_decay: float = 0.8
    scheduler_strategy: Literal[
        "none", "step", "cosine", "warmup_cosine"
    ] = "step"
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1
    minimum_learning_rate: float = 1e-6
    patience: int = 15
    report_every: int = 5
    random_seed: int = 42
    class_balance_strategy: Literal[
        "none", "weighted_loss", "balanced_sampler"
    ] = "none"
    gradient_clip_norm: float | None = 1.0


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    training_accuracy: tuple[float, ...]
    validation_accuracy: tuple[float, ...]
    training_macro_f1: tuple[float, ...]
    validation_macro_f1: tuple[float, ...]
    learning_rate: tuple[float, ...]
    best_epoch: int
    config: TrainingConfig | None = None


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics exposed after each epoch for monitoring and trial pruning."""

    epoch: int
    training_loss: float
    validation_loss: float
    training_accuracy: float
    validation_accuracy: float
    training_macro_f1: float
    validation_macro_f1: float
    learning_rate: float


EpochCallback = Callable[[EpochMetrics], None]


def train(
    model: nn.Module,
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    config: TrainingConfig | None = None,
    device: torch.device | str | None = None,
    epoch_callback: EpochCallback | None = None,
) -> TrainingHistory:
    """Train with fixed folds and restore the minimum-validation-loss weights."""
    cfg = config or TrainingConfig()
    _validate(
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        cfg,
    )
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    training_loader, validation_loader = _data_loaders(
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        cfg,
        pin_memory=resolved_device.type == "cuda",
    )
    model.to(resolved_device)
    training_loss_function = nn.CrossEntropyLoss(
        weight=_class_weights(training_targets, resolved_device)
        if cfg.class_balance_strategy == "weighted_loss"
        else None
    )
    validation_loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _learning_rate_scheduler(optimizer, cfg)

    training_losses: list[float] = []
    validation_losses: list[float] = []
    training_accuracies: list[float] = []
    validation_accuracies: list[float] = []
    training_macro_f1_scores: list[float] = []
    validation_macro_f1_scores: list[float] = []
    learning_rates: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_accuracy, train_macro_f1 = _run_epoch(
            model,
            training_loader,
            training_loss_function,
            resolved_device,
            optimizer,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )
        validation_loss, validation_accuracy, validation_macro_f1 = _run_epoch(
            model, validation_loader, validation_loss_function, resolved_device
        )
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_values = (
            train_loss,
            validation_loss,
            train_accuracy,
            validation_accuracy,
            train_macro_f1,
            validation_macro_f1,
            current_learning_rate,
        )
        if not all(isfinite(value) for value in epoch_values):
            raise FloatingPointError(
                f"Non-finite training metric encountered at epoch {epoch}: "
                f"{epoch_values}"
            )
        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        training_macro_f1_scores.append(train_macro_f1)
        validation_macro_f1_scores.append(validation_macro_f1)
        learning_rates.append(current_learning_rate)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch_callback is not None:
            epoch_callback(
                EpochMetrics(
                    epoch=epoch,
                    training_loss=train_loss,
                    validation_loss=validation_loss,
                    training_accuracy=train_accuracy,
                    validation_accuracy=validation_accuracy,
                    training_macro_f1=train_macro_f1,
                    validation_macro_f1=validation_macro_f1,
                    learning_rate=current_learning_rate,
                )
            )
        if scheduler is not None:
            scheduler.step()

        if epoch == 1 or epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:3d}: train loss {train_loss:.5f}, "
                f"validation loss {validation_loss:.5f}, "
                f"train accuracy {train_accuracy:.1%}, "
                f"validation accuracy {validation_accuracy:.1%}"
            )
        if stale_epochs >= cfg.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return TrainingHistory(
        training_loss=tuple(training_losses),
        validation_loss=tuple(validation_losses),
        training_accuracy=tuple(training_accuracies),
        validation_accuracy=tuple(validation_accuracies),
        training_macro_f1=tuple(training_macro_f1_scores),
        validation_macro_f1=tuple(validation_macro_f1_scores),
        learning_rate=tuple(learning_rates),
        best_epoch=best_epoch,
        config=cfg,
    )


def _validate(
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    config: TrainingConfig,
) -> None:
    _validate_dataset(training_features, training_targets, "training")
    _validate_dataset(validation_features, validation_targets, "validation")
    if training_features.shape[1:] != validation_features.shape[1:]:
        raise ValueError("training and validation features must have matching shapes.")
    if (
        config.epochs < 1
        or config.batch_size < 1
        or config.patience < 1
        or config.report_every < 1
        or config.scheduler_step_size < 1
        or config.warmup_epochs < 0
    ):
        raise ValueError(
            "epochs, batch_size, patience, report_every and scheduler_step_size "
            "must be positive; warmup_epochs must be non-negative."
        )
    if (
        config.learning_rate <= 0
        or config.weight_decay < 0
        or not 0 < config.learning_rate_decay <= 1
        or not 0 < config.warmup_start_factor <= 1
        or config.minimum_learning_rate < 0
    ):
        raise ValueError(
            "learning_rate must be positive, weight_decay and minimum_learning_rate "
            "non-negative, and decay/start factors in (0, 1]."
        )
    if config.minimum_learning_rate >= config.learning_rate:
        raise ValueError("minimum_learning_rate must be smaller than learning_rate.")
    if config.scheduler_strategy not in {
        "none",
        "step",
        "cosine",
        "warmup_cosine",
    }:
        raise ValueError("Unknown scheduler_strategy.")
    if (
        config.scheduler_strategy == "warmup_cosine"
        and not 0 < config.warmup_epochs < config.epochs
    ):
        raise ValueError(
            "warmup_cosine requires warmup_epochs between 1 and epochs - 1."
        )
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None.")
    if config.class_balance_strategy not in {
        "none",
        "weighted_loss",
        "balanced_sampler",
    }:
        raise ValueError("Unknown class_balance_strategy.")


def _validate_dataset(features: Tensor, targets: Tensor, name: str) -> None:
    if features.ndim < 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError(
            f"{name} features and one-dimensional targets must have equal length."
        )
    if (
        len(features) < 2
        or targets.is_floating_point()
        or torch.min(targets).item() < 0
    ):
        raise ValueError(
            f"{name} targets must be zero-based integers with at least two samples."
        )


def _learning_rate_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Construct the configured epoch-level learning-rate schedule."""
    if config.scheduler_strategy == "none":
        return None
    if config.scheduler_strategy == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.learning_rate_decay,
        )
    if config.scheduler_strategy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.minimum_learning_rate,
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.minimum_learning_rate,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.warmup_epochs],
    )


def _data_loaders(
    training_features: Tensor,
    training_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    config: TrainingConfig,
    *,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(config.random_seed)
    sampler = None
    shuffle = True
    if config.class_balance_strategy == "balanced_sampler":
        sample_weights = _sample_weights(training_targets)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return (
        DataLoader(
            TensorDataset(training_features.float(), training_targets.long()),
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            generator=generator,
            pin_memory=pin_memory,
        ),
        DataLoader(
            TensorDataset(validation_features.float(), validation_targets.long()),
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        ),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    *,
    gradient_clip_norm: float | None = None,
) -> tuple[float, float, float]:
    learning = optimizer is not None
    model.train(learning)
    total_loss = 0.0
    correct = 0
    sample_count = 0
    all_targets: list[Tensor] = []
    all_predictions: list[Tensor] = []
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
        total_loss += loss.item() * len(batch_targets)
        predictions = logits.argmax(1)
        correct += (predictions == batch_targets).sum().item()
        sample_count += len(batch_targets)
        all_targets.append(batch_targets.detach().cpu())
        all_predictions.append(predictions.detach().cpu())
    macro_f1 = f1_score(
        torch.cat(all_targets).numpy(),
        torch.cat(all_predictions).numpy(),
        average="macro",
        zero_division=0,
    )
    return total_loss / sample_count, correct / sample_count, float(macro_f1)


def _class_weights(targets: Tensor, device: torch.device) -> Tensor:
    counts = torch.bincount(targets.long())
    if torch.any(counts == 0):
        raise ValueError("weighted_loss requires every class in the training fold.")
    weights = counts.sum() / (len(counts) * counts.float())
    return weights.to(device)


def _sample_weights(targets: Tensor) -> Tensor:
    counts = torch.bincount(targets.long())
    if torch.any(counts == 0):
        raise ValueError(
            "balanced_sampler requires every class in the training fold."
        )
    return (1.0 / counts.float())[targets.long()]
