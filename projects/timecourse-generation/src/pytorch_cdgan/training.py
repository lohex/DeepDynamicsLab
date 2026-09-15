"""Function-based GAN training for biological trajectories."""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True, slots=True)
class GanTrainingConfig:
    epochs: int = 2_000
    batch_size: int = 256
    noise_size: int = 16
    learning_rate: float = 1e-4
    l2_penalty: float = 1e-4
    snapshot_every: int = 100
    report_every: int = 50
    convergence_tolerance: float = 0.05
    convergence_patience: int = 100
    random_seed: int = 32


@dataclass(frozen=True, slots=True)
class GanTrainingHistory:
    discriminator_real_loss: tuple[float, ...]
    discriminator_fake_loss: tuple[float, ...]
    generator_loss: tuple[float, ...]
    real_score: tuple[float, ...]
    fake_score: tuple[float, ...]
    snapshots: tuple[Tensor, ...]


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    trajectories: NDArray[np.floating] | Tensor,
    *,
    config: GanTrainingConfig | None = None,
    condition_labels: NDArray[np.integer] | Tensor | None = None,
    device: torch.device | str | None = None,
) -> GanTrainingHistory:
    """Train both GAN components and return scalar histories plus snapshots."""
    cfg = config or GanTrainingConfig()
    generator_noise_size = getattr(generator, "noise_size", None)
    if generator_noise_size is not None and generator_noise_size != cfg.noise_size:
        raise ValueError(
            "Generator noise_size and GanTrainingConfig.noise_size must match: "
            f"{generator_noise_size} != {cfg.noise_size}."
        )
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensor = torch.as_tensor(trajectories, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError("trajectories must have shape (samples, time) or (samples, 1, time).")
    labels_tensor = None
    if condition_labels is not None:
        labels_tensor = torch.as_tensor(condition_labels, dtype=torch.long)
        if labels_tensor.ndim != 1 or len(labels_tensor) != len(tensor):
            raise ValueError("condition_labels must have shape (samples,).")
    generator_is_conditional = getattr(generator, "condition_embedding", None) is not None
    discriminator_is_conditional = getattr(discriminator, "condition_embedding", None) is not None
    if generator_is_conditional != discriminator_is_conditional:
        raise ValueError("Generator and discriminator must either both be conditional or both unconditional.")
    if generator_is_conditional != (labels_tensor is not None):
        raise ValueError("Conditional models require condition_labels; unconditional models do not.")

    torch.manual_seed(cfg.random_seed)
    loader = DataLoader(
        TensorDataset(tensor) if labels_tensor is None else TensorDataset(tensor, labels_tensor),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
    generator.to(resolved_device)
    discriminator.to(resolved_device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCELoss()

    real_losses: list[float] = []
    fake_losses: list[float] = []
    generator_losses: list[float] = []
    real_scores: list[float] = []
    fake_scores: list[float] = []
    snapshots: list[Tensor] = []
    stable_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_metrics = np.zeros(5, dtype=float)
        sample_count = 0
        for batch in loader:
            real_batch = batch[0].to(resolved_device)
            condition_batch = batch[1].to(resolved_device) if labels_tensor is not None else None
            real_batch = real_batch.to(resolved_device)
            batch_size = len(real_batch)
            real_targets = torch.ones((batch_size, 1), device=resolved_device)
            fake_targets = torch.zeros((batch_size, 1), device=resolved_device)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_output = discriminator(real_batch, condition_batch)
            noise = torch.randn(batch_size, cfg.noise_size, device=resolved_device)
            fake_batch = generator(noise, condition_batch).detach()
            fake_output = discriminator(fake_batch, condition_batch)
            real_loss = criterion(real_output, real_targets)
            fake_loss = criterion(fake_output, fake_targets)
            regularization = sum(parameter.square().sum() for parameter in discriminator.parameters())
            discriminator_loss = real_loss + fake_loss + cfg.l2_penalty * regularization
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, cfg.noise_size, device=resolved_device)
            generated = generator(noise, condition_batch)
            generated_output = discriminator(generated, condition_batch)
            generator_loss = criterion(generated_output, real_targets)
            generator_loss.backward()
            generator_optimizer.step()

            epoch_metrics += np.array(
                [real_loss.item(), fake_loss.item(), generator_loss.item(), real_output.mean().item(), fake_output.mean().item()]
            ) * batch_size
            sample_count += batch_size

        epoch_metrics /= sample_count
        real_losses.append(float(epoch_metrics[0]))
        fake_losses.append(float(epoch_metrics[1]))
        generator_losses.append(float(epoch_metrics[2]))
        real_scores.append(float(epoch_metrics[3]))
        fake_scores.append(float(epoch_metrics[4]))

        if epoch == 1 or epoch % cfg.snapshot_every == 0:
            generator.eval()
            with torch.no_grad():
                snapshot_conditions = (
                    torch.arange(8, device=resolved_device)
                    % generator.num_conditions
                    if generator_is_conditional
                    else None
                )
                snapshots.append(
                    generator(torch.randn(8, cfg.noise_size, device=resolved_device), snapshot_conditions).cpu()
                )
            generator.train()
        if epoch == 1 or epoch % cfg.report_every == 0:
            print(
                f"epoch {epoch:4d}: D(real)={real_scores[-1]:.3f}, "
                f"D(fake)={fake_scores[-1]:.3f}, G loss={generator_losses[-1]:.3f}"
            )

        stable = abs(real_scores[-1] - 0.5) < cfg.convergence_tolerance
        stable_epochs = stable_epochs + 1 if stable else 0
        if stable_epochs >= cfg.convergence_patience:
            print(f"Stopping after {epoch} epochs with stable discriminator scores.")
            break

    return GanTrainingHistory(
        tuple(real_losses), tuple(fake_losses), tuple(generator_losses),
        tuple(real_scores), tuple(fake_scores), tuple(snapshots)
    )
