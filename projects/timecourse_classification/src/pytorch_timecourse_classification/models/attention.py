"""Self-attention classifier with sinusoidal positional encoding."""

import math
from typing import Literal

import torch
from torch import Tensor, nn


class AttentionClassifier(nn.Module):
    """Classify trajectories using self-attention without convolutions."""

    model_name = "attention"

    def __init__(
        self,
        *,
        input_length: int = 289,
        num_classes: int = 6,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        pooling: Literal["mean", "cls"] = "mean",
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        if input_length < 2:
            raise ValueError("input_length must be at least 2.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if (
            embedding_dim < 2
            or num_heads < 1
            or embedding_dim % 2
            or embedding_dim % num_heads
        ):
            raise ValueError(
                "embedding_dim must be even and divisible by a positive num_heads."
            )
        if num_layers < 1 or feedforward_dim < 1:
            raise ValueError("num_layers and feedforward_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'.")

        self.input_length = input_length
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.pooling = pooling
        self.norm_first = norm_first

        self.input_projection = nn.Linear(1, embedding_dim)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embedding_dim))
            if pooling == "cls"
            else None
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=input_length + int(pooling == "cls"),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    @property
    def model_config(self) -> dict[str, object]:
        """Return the constructor arguments needed to rebuild the model."""
        return {
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "norm_first": self.norm_first,
        }

    def forward(self, timecourses: Tensor) -> Tensor:
        if timecourses.ndim != 3 or timecourses.shape[1] != 1:
            raise ValueError("timecourses must have shape (batch, 1, time).")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(f"Expected {self.input_length} timepoints.")

        sequence = timecourses.transpose(1, 2)
        sequence = self.input_projection(sequence) * math.sqrt(self.embedding_dim)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(len(sequence), -1, -1)
            sequence = torch.cat((cls_token, sequence), dim=1)
        sequence = self.positional_encoding(sequence)
        encoded = self.encoder(sequence)
        pooled = encoded[:, 0] if self.pooling == "cls" else encoded.mean(dim=1)
        return self.classifier(pooled)

    def forward_with_attention(
        self, timecourses: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return logits and per-head attention for interpretation.

        Attention has shape ``(batch, layers, heads, tokens, tokens)``. A CLS
        model has one additional token at index zero. The hooks are temporary,
        so ordinary forward passes and saved parameters are unchanged.
        """
        captured_attention: list[Tensor] = []

        def request_attention_weights(
            module: nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return args, kwargs

        def capture_attention_weights(
            module: nn.Module,
            args: tuple[object, ...],
            output: tuple[Tensor, Tensor],
        ) -> None:
            captured_attention.append(output[1])

        handles = []
        for layer in self.encoder.layers:
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    request_attention_weights,
                    with_kwargs=True,
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(capture_attention_weights)
            )
        try:
            logits = self(timecourses)
        finally:
            for handle in handles:
                handle.remove()

        if len(captured_attention) != self.num_layers:
            raise RuntimeError("Could not capture attention from every encoder layer.")
        return logits, torch.stack(captured_attention, dim=1)


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sine/cosine positions to a batch-first sequence."""

    def __init__(self, *, embedding_dim: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, sequence: Tensor) -> Tensor:
        return sequence + self.encoding[:, : sequence.shape[1]]
