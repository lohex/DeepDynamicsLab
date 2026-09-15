"""Generator and discriminator for fixed-length signaling trajectories."""

import torch
from torch import Tensor, nn


class Discriminator(nn.Module):
    def __init__(
        self,
        input_length: int = 289,
        dropout: float = 0.3,
        num_conditions: int | None = None,
        condition_dim: int = 8,
    ) -> None:
        super().__init__()
        if num_conditions is not None and num_conditions < 1:
            raise ValueError("num_conditions must be positive.")
        if condition_dim < 1:
            raise ValueError("condition_dim must be positive.")
        self.num_conditions = num_conditions
        self.condition_dim = condition_dim
        self.condition_embedding = (
            nn.Embedding(num_conditions, condition_dim)
            if num_conditions is not None
            else None
        )
        # The condition embedding is broadcast over time as additional channels
        # before Flatten, hence each embedding component contributes input_length
        # values to the first linear layer.
        input_width = input_length * (1 + (condition_dim if num_conditions is not None else 0))
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_width, input_length * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_length * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, trajectories: Tensor, condition: Tensor | None = None) -> Tensor:
        if self.condition_embedding is not None:
            if condition is None:
                raise ValueError("A condition label is required by this discriminator.")
            condition_features = self.condition_embedding(condition.long())
            condition_features = condition_features.unsqueeze(-1).expand(
                -1, -1, trajectories.shape[-1]
            )
            trajectories = torch.cat([trajectories, condition_features], dim=1)
        elif condition is not None:
            raise ValueError("This discriminator is unconditional; condition must be None.")
        return self.network(trajectories)


class Generator(nn.Module):
    def __init__(
        self,
        noise_size: int = 16,
        output_length: int = 289,
        coarse_length: int | None = None,
        num_conditions: int | None = None,
        condition_dim: int = 8,
    ) -> None:
        super().__init__()
        if noise_size < 1:
            raise ValueError("noise_size must be positive.")
        if output_length < 2:
            raise ValueError("output_length must be at least 2.")
        if coarse_length is not None and not 2 <= coarse_length <= output_length:
            raise ValueError(
                "coarse_length must be between 2 and output_length."
            )
        self.noise_size = noise_size
        self.output_length = output_length
        self.coarse_length = coarse_length
        if num_conditions is not None and num_conditions < 1:
            raise ValueError("num_conditions must be positive.")
        if condition_dim < 1:
            raise ValueError("condition_dim must be positive.")
        self.num_conditions = num_conditions
        self.condition_dim = condition_dim
        self.condition_embedding = (
            nn.Embedding(num_conditions, condition_dim)
            if num_conditions is not None
            else None
        )
        input_noise_size = noise_size + (condition_dim if num_conditions is not None else 0)
        if coarse_length is None:
            self.network = nn.Sequential(
                nn.Linear(input_noise_size, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, output_length),
                nn.Unflatten(1, (1, output_length)),
                nn.Conv1d(1, 32, 5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 16, 7, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 1, 11, padding="same"),
            )
        else:
            # The dense network predicts a coarse control-point sequence.
            # Upsampling gives the discriminator the original time resolution;
            # the following convolutions can learn smooth local corrections.
            self.network = nn.Sequential(
                nn.Linear(input_noise_size, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 32 * coarse_length),
                nn.Unflatten(1, (32, coarse_length)),
                nn.Upsample(
                    size=output_length, mode="linear", align_corners=False
                ),
                nn.Conv1d(32, 32, 9, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 16, 7, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 1, 11, padding="same"),
            )

    def forward(self, noise: Tensor, condition: Tensor | None = None) -> Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_size:
            raise ValueError(
                f"noise must have shape (batch, {self.noise_size}), "
                f"got {tuple(noise.shape)}."
            )
        if self.condition_embedding is not None:
            if condition is None:
                raise ValueError("A condition label is required by this generator.")
            noise = torch.cat([noise, self.condition_embedding(condition.long())], dim=1)
        elif condition is not None:
            raise ValueError("This generator is unconditional; condition must be None.")
        return self.network(noise)


def initialize_weights(module: nn.Module) -> None:
    """Initialize layers with variance determined by their fan-in.

    The previous fixed ``Normal(0, 0.5)`` scale ignored how many inputs each
    unit receives. In this generator that caused variance to compound through
    several dense and convolutional layers, producing enormous initial
    trajectories. Kaiming normal initialization uses approximately
    ``sqrt(2 / fan_in)`` for ReLU layers, so the activation scale is kept
    comparable from one layer to the next.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        # The last generator convolution has no following ReLU. Use a unit
        # gain there; hidden layers are followed by ReLU and use sqrt(2).
        output_projection = isinstance(module, nn.Conv1d) and module.out_channels == 1
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_in",
            nonlinearity="linear" if output_projection else "relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
