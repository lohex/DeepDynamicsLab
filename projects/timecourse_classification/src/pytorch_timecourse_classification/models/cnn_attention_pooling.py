"""Convolutional token encoder with learned attention pooling."""

import torch
from torch import Tensor, nn


class CNNAttentionPoolingClassifier(nn.Module):
    """Extract local CNN features and learn which temporal tokens to pool."""

    model_name = "cnn_attention_pooling"

    def __init__(
        self,
        *,
        input_length: int = 289,
        num_classes: int = 6,
        channels: tuple[int, ...] = (32, 64, 96),
        kernel_sizes: tuple[int, ...] = (9, 7, 5),
        strides: tuple[int, ...] = (2, 2, 1),
        attention_hidden_dim: int = 64,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if input_length < 8:
            raise ValueError("input_length must be at least 8.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if not channels or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        if len(kernel_sizes) != len(channels) or len(strides) != len(channels):
            raise ValueError("kernel_sizes and strides must match channels.")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers.")
        if any(stride < 1 for stride in strides):
            raise ValueError("strides must contain positive integers.")
        if attention_hidden_dim < 1 or classifier_hidden_dim < 1:
            raise ValueError("hidden dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_length = input_length
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)
        self.attention_hidden_dim = attention_hidden_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.dropout = dropout

        widths = (1, *self.channels)
        encoder_layers: list[nn.Module] = []
        for index, (kernel_size, stride) in enumerate(
            zip(self.kernel_sizes, self.strides)
        ):
            encoder_layers.extend(
                (
                    nn.Conv1d(
                        widths[index],
                        widths[index + 1],
                        kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(widths[index + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.attention_score = nn.Sequential(
            nn.Linear(self.channels[-1], attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.channels[-1]),
            nn.Linear(self.channels[-1], classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    @property
    def model_config(self) -> dict[str, object]:
        """Return all constructor arguments needed to rebuild the model."""
        return {
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "channels": self.channels,
            "kernel_sizes": self.kernel_sizes,
            "strides": self.strides,
            "attention_hidden_dim": self.attention_hidden_dim,
            "classifier_hidden_dim": self.classifier_hidden_dim,
            "dropout": self.dropout,
        }

    def forward(self, timecourses: Tensor) -> Tensor:
        logits, _ = self.forward_with_attention(timecourses)
        return logits

    def forward_with_attention(self, timecourses: Tensor) -> tuple[Tensor, Tensor]:
        """Return class logits and pooling weights over downsampled CNN tokens."""
        self._validate_input(timecourses)
        tokens = self.encoder(timecourses).transpose(1, 2)
        attention = self.attention_score(tokens).squeeze(-1).softmax(dim=1)
        pooled = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
        return self.classifier(pooled), attention

    def _validate_input(self, timecourses: Tensor) -> None:
        if timecourses.ndim != 3 or timecourses.shape[1] != 1:
            raise ValueError("timecourses must have shape (batch, 1, time).")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(f"Expected {self.input_length} timepoints.")
