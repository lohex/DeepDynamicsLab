"""Length-preserving neural architectures for temporal burst segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemporalResidualBlock(nn.Module):
    """Two same-padded dilated convolutions with a residual connection."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        group_count = min(8, channels)
        while channels % group_count:
            group_count -= 1
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(group_count, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(group_count, channels),
        )
        self.activation = nn.GELU()

    def forward(self, sequence: Tensor) -> Tensor:
        return self.activation(sequence + self.layers(sequence))


class BurstTCN(nn.Module):
    """Predict one burst logit per input time point without pooling."""

    model_name = "burst_tcn"

    def __init__(
        self,
        *,
        input_channels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels < 1 or hidden_channels < 1:
            raise ValueError("input_channels and hidden_channels must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not dilations or any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must contain positive integers.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.dropout = dropout
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(
                TemporalResidualBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            )
        )
        self.output_head = nn.Conv1d(hidden_channels, 1, 1)

    @property
    def receptive_field(self) -> int:
        """The theoretical receptive field in samples."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
        }

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"features must have shape (batch, {self.input_channels}, time)."
            )
        encoded = self.blocks(self.input_projection(features))
        return self.output_head(encoded).squeeze(1)


class BurstCNN(nn.Module):
    """Length-preserving local CNN baseline without dilation or residual skips."""

    model_name = "burst_cnn"

    def __init__(
        self,
        *,
        input_channels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 9,
        depth: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels < 1 or hidden_channels < 1 or depth < 1:
            raise ValueError("input_channels, hidden_channels and depth must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout = dropout
        group_count = _group_count(hidden_channels)
        layers: list[nn.Module] = [
            nn.Conv1d(input_channels, hidden_channels, 1),
            nn.GroupNorm(group_count, hidden_channels),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.extend(
                (
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.GroupNorm(group_count, hidden_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.features = nn.Sequential(*layers)
        self.output_head = nn.Conv1d(hidden_channels, 1, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + self.depth * (self.kernel_size - 1)

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "depth": self.depth,
            "dropout": self.dropout,
        }

    def forward(self, features: Tensor) -> Tensor:
        _validate_features(features, self.input_channels)
        return self.output_head(self.features(features)).squeeze(1)


class BurstBiGRU(nn.Module):
    """Bidirectional recurrent baseline with one logit per time point."""

    model_name = "burst_bigru"

    def __init__(
        self,
        *,
        input_channels: int = 2,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_channels, hidden_size and num_layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.recurrent = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(2 * hidden_size, 1)

    @property
    def receptive_field(self) -> int | None:
        """A bidirectional recurrent state has no fixed finite local field."""
        return None

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def forward(self, features: Tensor) -> Tensor:
        _validate_features(features, self.input_channels)
        encoded, _ = self.recurrent(features.transpose(1, 2))
        return self.output_head(encoded).squeeze(-1)


class _UNetConvBlock(nn.Sequential):
    """Two same-padded convolutions at one temporal resolution."""

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, dropout: float,
    ) -> None:
        layers: list[nn.Module] = []
        for channels in (input_channels, output_channels):
            layers.extend((
                nn.Conv1d(channels, output_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(1, output_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        super().__init__(*layers)


class BurstUNet(nn.Module):
    """Offline 1D U-Net with concatenated skips and input-aligned burst logits.

    ``depth`` pooling stages require at least ``2**depth`` input samples.
    Decoder interpolation targets each skip's exact length, including odd
    lengths such as 289, so labels never need resizing or cropping.
    """

    model_name = "burst_unet"

    def __init__(
        self,
        *,
        input_channels: int = 2,
        base_channels: int = 16,
        depth: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels < 1 or base_channels < 1 or depth < 1:
            raise ValueError("input_channels, base_channels and depth must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout
        widths = [base_channels * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList([
            _UNetConvBlock(
                input_channels if level == 0 else widths[level - 1],
                widths[level], kernel_size, dropout,
            )
            for level in range(depth)
        ])
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = _UNetConvBlock(widths[-2], widths[-1], kernel_size, dropout)
        self.decoders = nn.ModuleList([
            _UNetConvBlock(widths[level + 1] + widths[level], widths[level], kernel_size, dropout)
            for level in reversed(range(depth))
        ])
        self.output_head = nn.Conv1d(widths[0], 1, 1)

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
        }

    def forward(self, features: Tensor) -> Tensor:
        _validate_features(features, self.input_channels)
        if features.shape[-1] < 2**self.depth:
            raise ValueError(f"features need at least {2**self.depth} time points.")
        encoded = features
        skips: list[Tensor] = []
        for encoder in self.encoders:
            encoded = encoder(encoded)
            skips.append(encoded)
            encoded = self.pool(encoded)
        encoded = self.bottleneck(encoded)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            encoded = F.interpolate(
                encoded, size=skip.shape[-1], mode="linear", align_corners=False,
            )
            encoded = decoder(torch.cat((encoded, skip), dim=1))
        return self.output_head(encoded).squeeze(1)


BURST_MODEL_TYPES: dict[str, type[nn.Module]] = {
    BurstTCN.model_name: BurstTCN,
    BurstCNN.model_name: BurstCNN,
    BurstBiGRU.model_name: BurstBiGRU,
    BurstUNet.model_name: BurstUNet,
}


def create_burst_model(model_name: str, model_config: dict[str, object]) -> nn.Module:
    """Construct one supported dense burst-segmentation architecture."""
    try:
        model_type = BURST_MODEL_TYPES[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown burst model {model_name!r}.") from error
    return model_type(**model_config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _group_count(channels: int) -> int:
    group_count = min(8, channels)
    while channels % group_count:
        group_count -= 1
    return group_count


def _validate_features(features: Tensor, input_channels: int) -> None:
    if features.ndim != 3 or features.shape[1] != input_channels:
        raise ValueError(
            f"features must have shape (batch, {input_channels}, time)."
        )
