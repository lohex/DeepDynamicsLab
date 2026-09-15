"""Attribution utilities for trained one-dimensional time-course classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ExplainableLayer:
    """A named convolutional layer and the domain of its activation axis."""

    name: str
    module: nn.Module
    domain: Literal["time", "frequency"]


def explainable_layers(model: nn.Module) -> tuple[ExplainableLayer, ...]:
    """Resolve meaningful final Conv1d layers of an Extended CNN."""
    architecture = getattr(model, "architecture", None)
    if not isinstance(architecture, str):
        raise TypeError("explainable_layers currently expects an Extended CNN.")
    layers: list[ExplainableLayer] = []
    if architecture in {"first_difference", "fft", "gated_fusion", "ordinal"}:
        raw_encoder = getattr(model, "raw_encoder", None)
        if raw_encoder is not None:
            layers.append(
                ExplainableLayer(
                    "raw",
                    raw_encoder[-3].convolution[0],
                    "time",
                )
            )
        derivative_encoder = getattr(model, "derivative_encoder", None)
        if derivative_encoder is not None:
            layers.append(
                ExplainableLayer(
                    "first_difference",
                    derivative_encoder[-3].convolution[0],
                    "time",
                )
            )
        fft_encoder = getattr(model, "fft_encoder", None)
        if fft_encoder is not None:
            layers.append(
                ExplainableLayer(
                    "fft_magnitude",
                    fft_encoder[-3].convolution[0],
                    "frequency",
                )
            )
    elif architecture == "multi_scale":
        encoder = model.specialized_encoder
        layers.extend(
            ExplainableLayer(f"scale_{index}", branch[0], "time")
            for index, branch in enumerate(encoder.branches)
        )
    elif architecture == "tcn":
        encoder = model.specialized_encoder
        layers.append(
            ExplainableLayer("tcn", encoder.blocks[-1].layers[4], "time")
        )
    elif architecture == "shift_robust":
        convolutions = [
            layer
            for layer in model.specialized_encoder.layers
            if isinstance(layer, nn.Conv1d)
        ]
        layers.append(ExplainableLayer("shift_robust", convolutions[-1], "time"))
    if not layers:
        raise ValueError(f"No explainable convolution found for {architecture!r}.")
    return tuple(layers)


def integrated_gradients(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    *,
    baselines: Tensor | None = None,
    steps: int = 64,
    internal_batch_size: int | None = None,
) -> Tensor:
    """Return input-level Integrated Gradients using bounded Captum batches."""
    if steps < 2:
        raise ValueError("steps must be at least two.")
    if len(inputs) < 1:
        raise ValueError("inputs must contain at least one sample.")
    resolved_internal_batch_size = (
        len(inputs) if internal_batch_size is None else internal_batch_size
    )
    if resolved_internal_batch_size < len(inputs):
        raise ValueError(
            "internal_batch_size must be at least the number of input samples."
        )
    IntegratedGradients = _captum_attr().IntegratedGradients
    model.eval()
    resolved_targets = _targets(model, inputs, targets)
    resolved_baselines = torch.zeros_like(inputs) if baselines is None else baselines
    return IntegratedGradients(model).attribute(
        inputs,
        baselines=resolved_baselines,
        target=resolved_targets,
        n_steps=steps,
        internal_batch_size=resolved_internal_batch_size,
    ).detach()


def input_saliency(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor | None = None,
) -> Tensor:
    """Return absolute input gradients as a deliberately simple reference."""
    Saliency = _captum_attr().Saliency
    model.eval()
    return Saliency(model).attribute(
        inputs,
        target=_targets(model, inputs, targets),
        abs=True,
    ).detach()


def layer_gradcam(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor | None,
    layer: ExplainableLayer,
    *,
    output_length: int | None = None,
) -> Tensor:
    """Return Grad-CAM for one temporal or spectral convolutional layer."""
    captum_attr = _captum_attr()
    model.eval()
    attribution = captum_attr.LayerGradCam(model, layer.module).attribute(
        inputs,
        target=_targets(model, inputs, targets),
        relu_attributions=True,
    )
    if output_length is not None:
        attribution = captum_attr.LayerAttribution.interpolate(
            attribution,
            (output_length,),
            interpolate_mode="linear",
        )
    return attribution.detach()


def temporal_occlusion(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    *,
    baselines: Tensor | None = None,
    window_size: int = 17,
    stride: int = 4,
) -> Tensor:
    """Map target-logit drops caused by masking overlapping temporal windows."""
    if inputs.ndim != 3 or inputs.shape[1] != 1:
        raise ValueError("inputs must have shape (batch, 1, time).")
    if window_size < 1 or window_size > inputs.shape[-1] or stride < 1:
        raise ValueError("window_size and stride must fit the input length.")
    model.eval()
    resolved_targets = _targets(model, inputs, targets)
    baseline = torch.zeros_like(inputs) if baselines is None else baselines
    if baseline.shape != inputs.shape:
        baseline = torch.broadcast_to(baseline, inputs.shape)
    with torch.no_grad():
        original = model(inputs).gather(1, resolved_targets[:, None]).squeeze(1)
    accumulated = torch.zeros_like(inputs)
    counts = torch.zeros_like(inputs)
    starts = list(range(0, inputs.shape[-1] - window_size + 1, stride))
    final_start = inputs.shape[-1] - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    with torch.no_grad():
        for start in starts:
            stop = start + window_size
            perturbed = inputs.clone()
            perturbed[..., start:stop] = baseline[..., start:stop]
            changed = model(perturbed).gather(
                1, resolved_targets[:, None]
            ).squeeze(1)
            drop = (original - changed)[:, None, None]
            accumulated[..., start:stop] += drop
            counts[..., start:stop] += 1
    return accumulated / counts.clamp_min(1)


def normalize_attributions(attributions: Tensor, *, absolute: bool = True) -> Tensor:
    """Scale every attribution map to [0, 1] without mixing samples."""
    resolved = attributions.abs() if absolute else attributions
    flattened = resolved.flatten(1)
    minimum = flattened.min(dim=1).values[:, None, None]
    maximum = flattened.max(dim=1).values[:, None, None]
    return (resolved - minimum) / (maximum - minimum).clamp_min(1e-12)


def temporal_window_deletion(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    attributions: Tensor,
    *,
    window_size: int = 17,
    baselines: Tensor | None = None,
    random_seed: int = 42,
) -> dict[str, Tensor]:
    """Compare target-logit drops for top-attribution and random windows."""
    if inputs.shape != attributions.shape or inputs.ndim != 3:
        raise ValueError("inputs and attributions must share shape (batch, 1, time).")
    if window_size < 1 or window_size > inputs.shape[-1]:
        raise ValueError("window_size must fit the input length.")
    model.eval()
    resolved_targets = targets.to(device=inputs.device, dtype=torch.long)
    baseline = torch.zeros_like(inputs) if baselines is None else baselines
    if baseline.shape != inputs.shape:
        baseline = torch.broadcast_to(baseline, inputs.shape)
    kernel = torch.ones(
        1, 1, window_size, device=inputs.device, dtype=inputs.dtype
    )
    window_mass = torch.conv1d(attributions.abs(), kernel).squeeze(1)
    top_starts = window_mass.argmax(dim=1)
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random_starts = torch.randint(
        0,
        inputs.shape[-1] - window_size + 1,
        (len(inputs),),
        generator=generator,
    ).to(inputs.device)
    with torch.no_grad():
        original = model(inputs).gather(1, resolved_targets[:, None]).squeeze(1)
        drops: dict[str, Tensor] = {}
        for name, starts in (("top", top_starts), ("random", random_starts)):
            perturbed = inputs.clone()
            for sample_index, start in enumerate(starts.tolist()):
                stop = start + window_size
                perturbed[sample_index, :, start:stop] = baseline[
                    sample_index, :, start:stop
                ]
            changed = model(perturbed).gather(
                1, resolved_targets[:, None]
            ).squeeze(1)
            drops[name] = (original - changed).detach()
    drops["top_start"] = top_starts.detach()
    drops["random_start"] = random_starts.detach()
    return drops


def _targets(model: nn.Module, inputs: Tensor, targets: Tensor | None) -> Tensor:
    if targets is not None:
        return targets.to(device=inputs.device, dtype=torch.long)
    with torch.no_grad():
        return model(inputs).argmax(dim=1)


def _captum_attr() -> Any:
    try:
        from captum import attr
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Captum is required for attribution. Install with "
            "`uv sync --extra torch --extra experiment`."
        ) from error
    return attr
