"""Shared visual analysis of model training histories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import Tensor, nn

from .data import FoldData
from .training import TrainingHistory


def summarize_history(history: TrainingHistory) -> dict[str, int | float]:
    """Return metrics at the checkpoint selected by validation loss."""
    best_index = history.best_epoch - 1
    return {
        "epochs_run": len(history.training_loss),
        "best_epoch": history.best_epoch,
        "training_loss": history.training_loss[best_index],
        "tune_loss": history.validation_loss[best_index],
        "training_accuracy": history.training_accuracy[best_index],
        "tune_accuracy": history.validation_accuracy[best_index],
        "training_macro_f1": history.training_macro_f1[best_index],
        "tune_macro_f1": history.validation_macro_f1[best_index],
        "accuracy_gap": (
            history.training_accuracy[best_index]
            - history.validation_accuracy[best_index]
        ),
        "macro_f1_gap": (
            history.training_macro_f1[best_index]
            - history.validation_macro_f1[best_index]
        ),
    }


def analyze_training_history(
    history: TrainingHistory,
    *,
    diagnose_fit: bool = False,
) -> dict[str, int | float]:
    """Print and plot the selected training state.

    ``diagnose_fit`` keeps the deliberately simple heuristic used by the CNN
    notebook. It is an orientation aid, not a statistical model comparison.
    """
    summary = summarize_history(history)
    print(summary)
    plot_training_history(history)
    if diagnose_fit:
        best_index = history.best_epoch - 1
        degradation = history.validation_loss[-1] - history.validation_loss[best_index]
        if summary["training_macro_f1"] < 0.50 and summary["tune_macro_f1"] < 0.50:
            print("Both train and tune macro-F1 are low: inspect for underfitting.")
        elif degradation > 0 and history.training_loss[-1] < history.training_loss[best_index]:
            print("Train loss improves after tune loss is best: inspect for overfitting.")
        else:
            print("No single strong under/overfitting signal; inspect the full curves.")
    return summary


def evaluate_classifier(
    model: nn.Module,
    fold: FoldData,
    *,
    class_names: tuple[str, ...],
    device: torch.device | str,
) -> dict[str, float]:
    """Evaluate one model on an explicitly supplied fold and show diagnostics."""
    labels = np.arange(len(class_names))
    predictions = predict_classes(model, fold, device=device)
    targets = fold.targets.cpu().numpy()
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    ConfusionMatrixDisplay.from_predictions(
        targets,
        predictions,
        labels=labels,
        display_labels=class_names,
        xticks_rotation=35,
        cmap="Blues",
    )
    plt.tight_layout()
    return classification_metrics(targets, predictions, report=report)


def predict_classes(
    model: nn.Module,
    fold: FoldData,
    *,
    device: torch.device | str,
    batch_size: int = 128,
) -> np.ndarray:
    """Return predictions using bounded-memory mini-batch inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model.eval()
    predictions: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(fold.features), batch_size):
            batch = fold.features[start:start + batch_size].to(device)
            predictions.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    report: dict[str, object] | None = None,
) -> dict[str, float]:
    """Return the scalar tune metrics shared by reports and parameter scans."""
    resolved_report = report or classification_report(
        targets,
        predictions,
        zero_division=0,
        output_dict=True,
    )
    macro_average = resolved_report["macro avg"]
    if not isinstance(macro_average, dict):
        raise TypeError("classification report has no macro-average mapping.")
    return {
        "tune_accuracy": float(np.mean(predictions == targets)),
        "tune_macro_f1": float(macro_average["f1-score"]),
    }


def plot_training_history(history: TrainingHistory) -> Figure:
    """Plot loss and accuracy with the restored epoch marked."""
    epochs = np.arange(1, len(history.training_loss) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, history.training_loss, label="train")
    axes[0].plot(epochs, history.validation_loss, label="tune")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].plot(epochs, history.training_accuracy, label="train")
    axes[1].plot(epochs, history.validation_accuracy, label="tune")
    axes[1].set(title="Accuracy", xlabel="epoch")
    for axis in axes:
        axis.axvline(
            history.best_epoch,
            color="black",
            linestyle=":",
            label="restored epoch",
        )
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_history_comparison(
    histories: dict[str, TrainingHistory],
) -> Figure:
    """Compare train/tune loss and accuracy with restored-epoch dots."""
    if not histories:
        raise ValueError("histories must contain at least one experiment.")
    panels = (
        ("training_loss", "Train loss", "loss"),
        ("validation_loss", "Tune loss", "loss"),
        ("training_accuracy", "Train accuracy", "accuracy"),
        ("validation_accuracy", "Tune accuracy", "accuracy"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = dict(zip(histories, plt.get_cmap("tab10").colors))
    for axis, (attribute, title, ylabel) in zip(axes.flat, panels):
        for experiment_name, history in histories.items():
            values = np.asarray(getattr(history, attribute))
            epochs = np.arange(1, len(values) + 1)
            color = colors[experiment_name]
            axis.plot(epochs, values, color=color, label=experiment_name)
            axis.scatter(
                history.best_epoch,
                values[history.best_epoch - 1],
                color=color,
                edgecolor="black",
                s=55,
                zorder=3,
            )
        axis.set(title=title, xlabel="epoch", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Training-history comparison")
    figure.tight_layout()
    return figure


def plot_ablation_diagnostics(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    title: str,
) -> Figure:
    """Plot performance, paired effects, convergence and generalization gaps."""
    if runs.empty or summary.empty:
        raise ValueError("runs and summary must not be empty.")
    order = list(summary.index)
    positions = np.arange(len(order))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    panels = (
        ("tune_macro_f1_mean", "tune_macro_f1_std", "Tune macro-F1", 0.0),
        ("macro_f1_delta_mean", "macro_f1_delta_std", "Paired Δ macro-F1", 0.0),
        ("best_epoch_mean", "best_epoch_std", "Selected epoch", None),
        ("macro_f1_gap_mean", None, "Train − tune macro-F1", 0.0),
    )
    for axis, (mean_column, std_column, panel_title, reference_line) in zip(
        axes.flat, panels
    ):
        means = summary.loc[order, mean_column].to_numpy(dtype=float)
        errors = (
            summary.loc[order, std_column].fillna(0).to_numpy(dtype=float)
            if std_column else None
        )
        axis.bar(positions, means, yerr=errors, capsize=3, color="tab:blue", alpha=0.75)
        run_column = {
            "tune_macro_f1_mean": "tune_macro_f1",
            "macro_f1_delta_mean": "tune_macro_f1_delta",
            "best_epoch_mean": "best_epoch",
            "macro_f1_gap_mean": "macro_f1_gap",
        }[mean_column]
        for position, name in enumerate(order):
            values = runs.loc[runs["name"] == name, run_column].dropna().to_numpy()
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(
                position + np.asarray(offsets), values,
                color="black", s=20, zorder=3,
            )
        if reference_line is not None:
            axis.axhline(reference_line, color="0.25", linewidth=1, linestyle=":")
        axis.set(
            title=panel_title,
            xticks=positions,
            xticklabels=order,
        )
        axis.tick_params(axis="x", labelrotation=55)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    return figure


def attention_rollout(attention: Tensor, *, pooling: str) -> Tensor:
    """Project multi-layer self-attention back to input timepoints.

    Heads are averaged and an identity residual is added before multiplying
    attention matrices across layers. The result is normalized per sample.
    """
    if attention.ndim != 5 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError(
            "attention must have shape (batch, layers, heads, tokens, tokens)."
        )
    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be 'mean' or 'cls'.")

    mean_attention = attention.mean(dim=2)
    token_count = mean_attention.shape[-1]
    identity = torch.eye(
        token_count,
        dtype=mean_attention.dtype,
        device=mean_attention.device,
    ).expand(len(mean_attention), -1, -1)
    joint_attention = identity
    for layer_attention in mean_attention.unbind(dim=1):
        layer_attention = layer_attention + identity
        layer_attention = layer_attention / layer_attention.sum(
            dim=-1, keepdim=True
        )
        joint_attention = layer_attention @ joint_attention

    if pooling == "cls":
        importance = joint_attention[:, 0, 1:]
    else:
        importance = joint_attention.mean(dim=1)
    return importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def plot_attention_overlay(
    trajectory: np.ndarray,
    importance: np.ndarray,
    *,
    axis: plt.Axes,
    title: str,
) -> None:
    """Plot one trajectory with timepoints colored by rollout importance."""
    if trajectory.ndim != 1 or importance.shape != trajectory.shape:
        raise ValueError("trajectory and importance must be equal-length vectors.")
    time = np.arange(len(trajectory))
    axis.plot(time, trajectory, color="0.65", linewidth=1, zorder=1)
    points = axis.scatter(
        time,
        trajectory,
        c=importance,
        cmap="magma",
        s=10,
        zorder=2,
    )
    axis.set(title=title, xlabel="time index", ylabel="signal")
    axis.figure.colorbar(points, ax=axis, label="attention rollout")
