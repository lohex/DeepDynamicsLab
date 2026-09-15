"""Extended 1D CNN architectures for biological time courses."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ExtendedCNNArchitecture = Literal[
    "first_difference",
    "fft",
    "gated_fusion",
    "multi_scale",
    "tcn",
    "shift_robust",
    "ordinal",
]


class ExtendedCNNClassifier(nn.Module):
    """Build one of several CNN inductive biases for a 1D time course."""

    model_name = "extended_cnn"
    supported_architectures = (
        "first_difference",
        "fft",
        "gated_fusion",
        "multi_scale",
        "tcn",
        "shift_robust",
        "ordinal",
    )

    def __init__(
        self,
        *,
        input_length: int = 289,
        num_classes: int = 6,
        architecture: ExtendedCNNArchitecture = "first_difference",
        branch_channels: tuple[int, ...] = (32, 64, 96),
        kernel_sizes: tuple[int, ...] = (9, 7, 5),
        dilations: tuple[int, ...] = (1, 2, 4),
        hidden_dims: tuple[int, ...] = (128, 32),
        dropout: float = 0.3,
        adaptive_pool_size: int = 4,
        multi_scale_kernel_sizes: tuple[int, ...] = (3, 7, 15),
        ordinal_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_length < 16:
            raise ValueError("input_length must be at least 16.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if architecture not in self.supported_architectures:
            raise ValueError(f"Unsupported architecture: {architecture!r}.")
        if not branch_channels or any(channel < 1 for channel in branch_channels):
            raise ValueError("branch_channels must contain positive integers.")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if len(kernel_sizes) != len(branch_channels):
            raise ValueError("kernel_sizes must match branch_channels.")
        if len(dilations) != len(branch_channels):
            raise ValueError("dilations must match branch_channels.")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers.")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must contain positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if adaptive_pool_size < 1:
            raise ValueError("adaptive_pool_size must be positive.")
        if not multi_scale_kernel_sizes or any(
            kernel < 1 or kernel % 2 == 0 for kernel in multi_scale_kernel_sizes
        ):
            raise ValueError(
                "multi_scale_kernel_sizes must contain positive odd integers."
            )
        if ordinal_temperature <= 0:
            raise ValueError("ordinal_temperature must be positive.")

        self.input_length = input_length
        self.num_classes = num_classes
        self.architecture = architecture
        self.branch_channels = tuple(branch_channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.dilations = tuple(dilations)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.adaptive_pool_size = adaptive_pool_size
        self.multi_scale_kernel_sizes = tuple(multi_scale_kernel_sizes)
        self.ordinal_temperature = ordinal_temperature

        branch_width = self.branch_channels[-1] * adaptive_pool_size
        self.raw_encoder: nn.Module | None = None
        self.derivative_encoder: nn.Module | None = None
        self.fft_encoder: nn.Module | None = None
        self.gate: nn.Module | None = None
        self.specialized_encoder: nn.Module | None = None

        if architecture in {"first_difference", "fft", "gated_fusion"}:
            self.raw_encoder = self._branch_encoder()
            self.derivative_encoder = self._branch_encoder()
            if architecture == "fft":
                self.fft_encoder = self._branch_encoder()
                feature_width = 3 * branch_width
            elif architecture == "gated_fusion":
                self.gate = nn.Sequential(
                    nn.Linear(2 * branch_width, branch_width), nn.Sigmoid()
                )
                feature_width = branch_width
            else:
                feature_width = 2 * branch_width
        elif architecture == "multi_scale":
            self.specialized_encoder = MultiScaleEncoder(
                output_channels=self.branch_channels[-1],
                kernel_sizes=self.multi_scale_kernel_sizes,
                dropout=dropout,
            )
            feature_width = self.branch_channels[-1] * len(
                self.multi_scale_kernel_sizes
            )
        elif architecture == "tcn":
            self.specialized_encoder = TemporalConvolutionalEncoder(
                width=self.branch_channels[-1],
                dilations=self.dilations,
                kernel_size=self.kernel_sizes[0],
                dropout=dropout,
            )
            feature_width = self.branch_channels[-1]
        elif architecture == "shift_robust":
            self.specialized_encoder = ShiftRobustEncoder(
                channels=self.branch_channels,
                kernel_sizes=self.kernel_sizes,
                dropout=dropout,
            )
            feature_width = self.branch_channels[-1]
        else:
            self.raw_encoder = self._branch_encoder()
            feature_width = branch_width

        self.classifier: nn.Module
        if architecture == "ordinal":
            self.classifier = OrdinalHead(
                feature_width,
                hidden_dims=self.hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
                temperature=ordinal_temperature,
            )
        else:
            self.classifier = MLPHead(
                feature_width,
                hidden_dims=self.hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
            )

    @property
    def model_config(self) -> dict[str, object]:
        """Return all constructor arguments needed to rebuild the model."""
        return {
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "architecture": self.architecture,
            "branch_channels": self.branch_channels,
            "kernel_sizes": self.kernel_sizes,
            "dilations": self.dilations,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "adaptive_pool_size": self.adaptive_pool_size,
            "multi_scale_kernel_sizes": self.multi_scale_kernel_sizes,
            "ordinal_temperature": self.ordinal_temperature,
        }

    def forward(self, timecourses: Tensor) -> Tensor:
        self._validate_input(timecourses)
        if self.architecture in {"first_difference", "fft", "gated_fusion"}:
            assert self.raw_encoder is not None
            assert self.derivative_encoder is not None
            raw = self.raw_encoder(timecourses)
            derivative = self.derivative_encoder(torch.diff(timecourses, dim=2))
            if self.architecture == "gated_fusion":
                assert self.gate is not None
                gate = self.gate(torch.cat((raw, derivative), dim=1))
                features = gate * raw + (1.0 - gate) * derivative
            else:
                branches = [raw, derivative]
                if self.architecture == "fft":
                    assert self.fft_encoder is not None
                    spectrum = torch.log1p(
                        torch.abs(torch.fft.rfft(timecourses, dim=2))
                    )
                    branches.append(self.fft_encoder(spectrum))
                features = torch.cat(branches, dim=1)
        elif self.architecture == "ordinal":
            assert self.raw_encoder is not None
            features = self.raw_encoder(timecourses)
        else:
            assert self.specialized_encoder is not None
            features = self.specialized_encoder(timecourses)
        return self.classifier(features)

    def _branch_encoder(self) -> nn.Sequential:
        channels = (1, *self.branch_channels)
        layers: list[nn.Module] = []
        for index, (kernel_size, dilation) in enumerate(
            zip(self.kernel_sizes, self.dilations)
        ):
            layers.append(
                ResidualDownsamplingBlock(
                    channels[index],
                    channels[index + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )
        layers.extend((nn.AdaptiveAvgPool1d(self.adaptive_pool_size), nn.Flatten()))
        return nn.Sequential(*layers)

    def _validate_input(self, timecourses: Tensor) -> None:
        if timecourses.ndim != 3 or timecourses.shape[1] != 1:
            raise ValueError("timecourses must have shape (batch, 1, time).")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(f"Expected {self.input_length} timepoints.")


class ResidualDownsamplingBlock(nn.Module):
    """Dilated stride-two convolution with a projected residual path."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.convolution = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                stride=2,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(output_channels),
        )
        self.residual_projection = nn.Conv1d(
            input_channels, output_channels, 1, stride=2
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        residual = self.residual_projection(sequence)
        return self.dropout(self.activation(self.convolution(sequence) + residual))


class MultiScaleEncoder(nn.Module):
    """Parallel convolutional filters expose short, medium and long motifs."""

    def __init__(
        self,
        *,
        output_channels: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1, output_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(output_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            for kernel in kernel_sizes
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return torch.cat([branch(sequence) for branch in self.branches], dim=1)


class TemporalConvolutionalEncoder(nn.Module):
    """Residual, exponentially dilated convolutions with a wide receptive field."""

    def __init__(
        self,
        *,
        width: int,
        dilations: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(1, width, 1)
        self.blocks = nn.Sequential(
            *(
                TemporalResidualBlock(
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            )
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, sequence: Tensor) -> Tensor:
        return self.pool(self.blocks(self.input_projection(sequence)))


class TemporalResidualBlock(nn.Module):
    """Length-preserving residual TCN block."""

    def __init__(
        self,
        width: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.layers = nn.Sequential(
            nn.Conv1d(
                width, width, kernel_size, padding=padding, dilation=dilation
            ),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                width, width, kernel_size, padding=padding, dilation=dilation
            ),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.GELU()

    def forward(self, sequence: Tensor) -> Tensor:
        return self.activation(sequence + self.layers(sequence))


class ShiftRobustEncoder(nn.Module):
    """Preserve time resolution before global pooling to reduce shift sensitivity."""

    def __init__(
        self,
        *,
        channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        widths = (1, *channels)
        layers: list[nn.Module] = []
        for index, kernel_size in enumerate(kernel_sizes):
            layers.extend(
                (
                    nn.Conv1d(
                        widths[index],
                        widths[index + 1],
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(widths[index + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        layers.extend((nn.AdaptiveAvgPool1d(1), nn.Flatten()))
        self.layers = nn.Sequential(*layers)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.layers(sequence)


class MLPHead(nn.Sequential):
    """Regularized dense classification head."""

    def __init__(
        self,
        input_width: int,
        *,
        hidden_dims: tuple[int, ...],
        num_classes: int,
        dropout: float,
    ) -> None:
        layers: list[nn.Module] = []
        previous_width = input_width
        for width in hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous_width, width),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            previous_width = width
        layers.append(nn.Linear(previous_width, num_classes))
        super().__init__(*layers)


class OrdinalHead(nn.Module):
    """Map one latent dose score to logits around learned ordered class centers."""

    def __init__(
        self,
        input_width: int,
        *,
        hidden_dims: tuple[int, ...],
        num_classes: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.score = MLPHead(
            input_width,
            hidden_dims=hidden_dims,
            num_classes=1,
            dropout=dropout,
        )
        initial_gap = 2.0 / (num_classes - 1)
        inverse_softplus = math.log(math.expm1(initial_gap))
        self.first_center = nn.Parameter(torch.tensor(-1.0))
        self.raw_gaps = nn.Parameter(
            torch.full((num_classes - 1,), inverse_softplus)
        )
        self.temperature = temperature

    @property
    def ordered_centers(self) -> Tensor:
        gaps = F.softplus(self.raw_gaps)
        return torch.cat(
            (self.first_center.reshape(1), self.first_center + gaps.cumsum(0))
        )

    def forward(self, features: Tensor) -> Tensor:
        score = self.score(features)
        distances = score - self.ordered_centers.unsqueeze(0)
        return -(distances.square()) / self.temperature
