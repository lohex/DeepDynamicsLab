"""Shared orchestration for the model-family training notebooks."""

from collections.abc import Callable

import torch
from torch import nn

from .artifacts import (
    ArtifactPaths,
    artifact_dir_for,
    load_model,
    save_experiment_artifacts,
)
from .data import FoldData, Preprocessor
from .training import EpochCallback, TrainingConfig, TrainingHistory, train


ModelFactory = Callable[..., nn.Module]


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def create_model(
    model_type: ModelFactory,
    *,
    input_length: int,
    num_classes: int,
    random_seed: int = 42,
    **model_config: object,
) -> tuple[nn.Module, int]:
    """Seed, construct and describe one classifier variant."""
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    model = model_type(
        input_length=input_length,
        num_classes=num_classes,
        **model_config,
    )
    parameter_count = trainable_parameter_count(model)
    print("model_config:", getattr(model, "model_config", model_config))
    print(f"trainable parameters: {parameter_count:,}")
    return model, parameter_count


def fit_model(
    model: nn.Module,
    training_config: TrainingConfig,
    *,
    training_fold: FoldData,
    tuning_fold: FoldData,
    device: torch.device | str,
    epoch_callback: EpochCallback | None = None,
) -> TrainingHistory:
    """Train a model using explicit fixed train and tune folds."""
    print(f"training_config: {training_config}")
    return train(
        model,
        training_fold.features,
        training_fold.targets,
        tuning_fold.features,
        tuning_fold.targets,
        config=training_config,
        device=device,
        epoch_callback=epoch_callback,
    )


def save_and_verify_experiment(
    experiment_name: str,
    model: nn.Module,
    history: TrainingHistory,
    *,
    preprocessor: Preprocessor,
    training_fold: FoldData,
    tuning_fold: FoldData,
    device: torch.device | str,
) -> tuple[nn.Module, ArtifactPaths]:
    """Save an experiment, reload it and verify weights and output shape."""
    paths = save_experiment_artifacts(
        artifact_dir_for(experiment_name),
        model=model,
        history=history,
        preprocessor=preprocessor,
        training_targets=training_fold.targets,
        validation_targets=tuning_fold.targets,
        device=device,
    )
    restored_model, checkpoint = load_model(paths.model, device=device)
    for parameter_name, expected_value in model.state_dict().items():
        restored_value = restored_model.state_dict()[parameter_name]
        if not torch.equal(restored_value, expected_value):
            raise AssertionError(
                f"Reloaded parameter {parameter_name!r} differs from the saved model."
            )
    sample_count = min(16, len(tuning_fold.features))
    sample_features = tuning_fold.features[:sample_count].to(device)
    model.eval()
    with torch.no_grad():
        expected_logits = model(sample_features)
        restored_logits = restored_model(sample_features)
    expected_shape = (sample_count, len(preprocessor.class_names))
    if tuple(restored_logits.shape) != expected_shape:
        raise AssertionError(
            f"Reloaded model output has shape {tuple(restored_logits.shape)}, "
            f"expected {expected_shape}."
        )
    torch.testing.assert_close(
        restored_logits,
        expected_logits,
        rtol=0.0,
        atol=0.0,
        msg="Reloaded model logits differ from the saved model.",
    )
    print(paths)
    print(
        f"reloaded {checkpoint['model_name']} at best epoch "
        f"{checkpoint['best_epoch']}"
    )
    return restored_model, paths
