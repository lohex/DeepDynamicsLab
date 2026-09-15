# Generative AI for biological time courses

## Goal

Learn dose-conditioned distributions of single-cell signaling trajectories with GAN and VAE models. The central challenge is not only producing visually plausible curves, but preserving dose-specific dynamics, variability and roughness without collapsing to an average trajectory.

## ML engineering questions

- How can adversarial training be diagnosed when losses and discriminator scores alone do not measure sample quality?
- Does generating a full-resolution signal directly differ from predicting coarse control points and learning the upsampling corrections?
- How should layer initialization depend on fan-in so activation scales remain controlled at the start of training?
- Does conditioning both generator and discriminator produce samples with the requested dose characteristics?
- Which distributional and temporal statistics reveal excessive noise, oversmoothing or mode collapse?

## Model design

The introductory conditional GAN concatenates Gaussian latent noise with a learned dose embedding. Two generator variants are compared:

1. **Full-resolution generator:** a dense projection creates the complete trajectory before convolutional refinement.
2. **Coarse generator:** a configurable number of control points is predicted, linearly upsampled to the original length and refined by temporal convolutions.

The discriminator receives the trajectory together with a dose embedding broadcast over time. Linear and convolutional layers use fan-in-aware Kaiming initialization instead of a fixed weight standard deviation; this prevents variance from compounding through layers and creating unrealistically large initial signals.

## Contents

- [`notebooks/01_conditional_gan_generator_comparison.ipynb`](notebooks/01_conditional_gan_generator_comparison.ipynb): data inspection, direct and coarse generator variants, per-variant training diagnostics, dose-wise sample analysis and final comparison
- [`notebooks/02_classifier_guided_gans.ipynb`](notebooks/02_classifier_guided_gans.ipynb): three controlled classifier-guided GAN variants using a small CNN, the validation-winning Gated-Fusion model type from scratch, and the same model initialized with its saved classification weights
- [`notebooks/03_gated_fusion_wasserstein_vs_kl.ipynb`](notebooks/03_gated_fusion_wasserstein_vs_kl.ipynb): controlled comparison of WGAN-GP and a forward-KL f-GAN with the same frozen Gated-Fusion dose guide
- [`notebooks/04_conditional_vae.ipynb`](notebooks/04_conditional_vae.ipynb): classifier-guided conditional beta-VAE with KL warm-up, validation diagnostics and held-out dose-wise feature comparisons
- `src/pytorch_cdgan/model.py`: generator and discriminator
- `src/pytorch_cdgan/training.py`: function-based adversarial training loop

## Evaluation

The notebooks train only on the fixed train fold and use validation for model-development decisions. They compare held-out real data and generated samples separately for every dose using example trajectories and temporal features such as mean, standard deviation, maximum, dynamic range, endpoint change and mean absolute step. The final comparisons place real data and competing generative models side by side.

Discriminator outputs near 0.5 are only a training diagnostic. They are not sufficient evidence of realism, diversity or correct conditioning; feature distributions and dose-wise samples remain essential.

## Run

```bash
uv sync --extra torch
jupyter lab projects/timecourse-generation/notebooks/01_conditional_gan_generator_comparison.ipynb
jupyter lab projects/timecourse-generation/notebooks/02_classifier_guided_gans.ipynb
jupyter lab projects/timecourse-generation/notebooks/03_gated_fusion_wasserstein_vs_kl.ipynb
jupyter lab projects/timecourse-generation/notebooks/04_conditional_vae.ipynb
```
