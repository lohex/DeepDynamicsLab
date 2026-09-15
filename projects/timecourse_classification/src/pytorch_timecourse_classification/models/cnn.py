"""Convolutional classifier for single-cell time courses."""

from math import isfinite

from torch import Tensor, nn


BASE_CNN_CHANNELS = (32, 64, 128)
BASE_CNN_HIDDEN_DIM = 64


def scaled_cnn_config(width_multiplier: float) -> dict[str, object]:
    """Scale CNN channels and head width while keeping depth and kernels fixed."""
    if not isfinite(width_multiplier) or width_multiplier <= 0:
        raise ValueError("width_multiplier must be finite and positive.")
    channels = tuple(
        int(round(channel * width_multiplier))
        for channel in BASE_CNN_CHANNELS
    )
    hidden_dim = int(round(BASE_CNN_HIDDEN_DIM * width_multiplier))
    if min(*channels, hidden_dim) < 1:
        raise ValueError("width_multiplier produces a zero-width layer.")
    return {
        "channels": channels,
        "hidden_dim": hidden_dim,
    }


class CNNClassifier(nn.Module):
    """Extract local temporal motifs with a single convolutional stream."""

    model_name = "cnn"

    def __init__(
        self,
        *,
        input_length: int = 289,
        num_classes: int = 6,
        channels: tuple[int, ...] = BASE_CNN_CHANNELS,
        kernel_sizes: tuple[int, ...] = (9, 7, 5),
        dilations: tuple[int, ...] | None = None,
        pool_size: int = 2,
        hidden_dim: int = BASE_CNN_HIDDEN_DIM,
        dropout: float = 0.2,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if input_length < 16:
            raise ValueError("input_length must be at least 16.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if not channels or len(channels) != len(kernel_sizes):
            raise ValueError(
                "channels and kernel_sizes must have equal, non-zero length."
            )
        if any(channel < 1 for channel in channels):
            raise ValueError("all channel counts must be positive.")
        if any(
            kernel_size < 1 or kernel_size % 2 == 0
            for kernel_size in kernel_sizes
        ):
            raise ValueError("kernel sizes must be positive odd integers.")
        resolved_dilations = dilations or (1,) * len(channels)
        if len(resolved_dilations) != len(channels) or any(
            dilation < 1 for dilation in resolved_dilations
        ):
            raise ValueError("dilations must contain one positive value per block.")
        if pool_size < 1 or pool_size ** len(channels) > input_length:
            raise ValueError("pool_size must leave at least one temporal value.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_length = input_length
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.dilations = tuple(resolved_dilations)
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.residual = residual
        convolution_blocks: list[nn.Module] = []
        input_channels = 1
        for output_channels, kernel_size, dilation in zip(
            self.channels, self.kernel_sizes, self.dilations
        ):
            if residual:
                block: nn.Module = ResidualConvolutionBlock(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    pool_size=pool_size,
                    dropout=dropout,
                )
            else:
                block = _convolution_block(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    pool_size=pool_size,
                    dropout=dropout,
                )
            convolution_blocks.append(block)
            input_channels = output_channels
        self.features = nn.Sequential(
            *convolution_blocks,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.channels[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @property
    def model_config(self) -> dict[str, object]:
        """Return the constructor arguments needed to rebuild the model."""
        return {
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "channels": self.channels,
            "kernel_sizes": self.kernel_sizes,
            "dilations": self.dilations,
            "pool_size": self.pool_size,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "residual": self.residual,
        }

    def forward(self, timecourses: Tensor) -> Tensor:
        if timecourses.ndim != 3 or timecourses.shape[1] != 1:
            raise ValueError("timecourses must have shape (batch, 1, time).")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(f"Expected {self.input_length} timepoints.")
        return self.classifier(self.features(timecourses))


def _convolution_block(
    input_channels: int,
    output_channels: int,
    *,
    kernel_size: int,
    dilation: int,
    pool_size: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        ),
        nn.BatchNorm1d(output_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool_size),
        nn.Dropout(dropout),
    )


class ResidualConvolutionBlock(nn.Module):
    """Two dilated convolutions with a projected residual connection."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(output_channels),
        )
        self.residual_projection = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        residual = self.residual_projection(sequence)
        sequence = self.activation(self.convolutions(sequence) + residual)
        return self.dropout(self.pool(sequence))
