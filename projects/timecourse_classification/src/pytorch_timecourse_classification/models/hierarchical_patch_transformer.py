"""Hierarchical patch transformer for 1D time courses."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import SinusoidalPositionalEncoding


class HierarchicalPatchTransformerClassifier(nn.Module):
    """Encode local patches, merge neighbors, then model coarse interactions."""

    model_name = "hierarchical_patch_transformer"

    def __init__(
        self,
        *,
        input_length: int = 289,
        num_classes: int = 6,
        patch_size: int = 8,
        embedding_dim: int = 48,
        num_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 2,
        feedforward_multiplier: int = 3,
        dropout: float = 0.2,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if input_length < 8:
            raise ValueError("input_length must be at least 8.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if patch_size < 2:
            raise ValueError("patch_size must be at least 2.")
        if (
            embedding_dim < 2
            or embedding_dim % 2
            or embedding_dim % num_heads
        ):
            raise ValueError(
                "embedding_dim must be even and divisible by num_heads."
            )
        if stage1_layers < 1 or stage2_layers < 1:
            raise ValueError("Both stages must contain at least one layer.")
        if feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_length = input_length
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.stage1_layers = stage1_layers
        self.stage2_layers = stage2_layers
        self.feedforward_multiplier = feedforward_multiplier
        self.dropout = dropout
        self.norm_first = norm_first
        self.patch_count = math.ceil(input_length / patch_size)
        self.stage2_dim = 2 * embedding_dim

        self.patch_projection = nn.Conv1d(
            1, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.stage1_position = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim, max_length=self.patch_count
        )
        self.stage1 = self._encoder(
            embedding_dim, stage1_layers, embedding_dim * feedforward_multiplier
        )
        self.patch_merge = nn.Sequential(
            nn.LayerNorm(self.stage2_dim),
            nn.Linear(self.stage2_dim, self.stage2_dim),
        )
        self.stage2_position = SinusoidalPositionalEncoding(
            embedding_dim=self.stage2_dim,
            max_length=math.ceil(self.patch_count / 2),
        )
        self.stage2 = self._encoder(
            self.stage2_dim,
            stage2_layers,
            self.stage2_dim * feedforward_multiplier,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.stage2_dim),
            nn.Dropout(dropout),
            nn.Linear(self.stage2_dim, num_classes),
        )

    @property
    def model_config(self) -> dict[str, object]:
        """Return all constructor arguments needed to rebuild the model."""
        return {
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "stage1_layers": self.stage1_layers,
            "stage2_layers": self.stage2_layers,
            "feedforward_multiplier": self.feedforward_multiplier,
            "dropout": self.dropout,
            "norm_first": self.norm_first,
        }

    def forward(self, timecourses: Tensor) -> Tensor:
        self._validate_input(timecourses)
        padding = self.patch_count * self.patch_size - self.input_length
        if padding:
            timecourses = F.pad(timecourses, (0, padding), mode="replicate")
        tokens = self.patch_projection(timecourses).transpose(1, 2)
        tokens = self.stage1(self.stage1_position(tokens))
        if tokens.shape[1] % 2:
            tokens = torch.cat((tokens, tokens[:, -1:]), dim=1)
        tokens = tokens.reshape(
            len(tokens), tokens.shape[1] // 2, self.stage2_dim
        )
        tokens = self.patch_merge(tokens)
        tokens = self.stage2(self.stage2_position(tokens))
        return self.classifier(tokens.mean(dim=1))

    def _encoder(
        self, dimension: int, layers: int, feedforward_dim: int
    ) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=self.num_heads,
            dim_feedforward=feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=self.norm_first,
        )
        return nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )

    def _validate_input(self, timecourses: Tensor) -> None:
        if timecourses.ndim != 3 or timecourses.shape[1] != 1:
            raise ValueError("timecourses must have shape (batch, 1, time).")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(f"Expected {self.input_length} timepoints.")
