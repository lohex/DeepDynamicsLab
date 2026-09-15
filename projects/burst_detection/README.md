# Burst detection in single-cell trajectories

## Goal

Predict a burst probability at every one of 289 time points in a single-cell
SMAD trajectory. This is temporal segmentation, not whole-trajectory
classification: the model preserves sequence length and the evaluation asks
both whether individual points and complete biological events were recovered.

## Scientific starting point

The synthetic generator ports the structure of
[`stochastic-smad/+burstDetection/generateSample.m`](https://github.com/lohex/stochastic-smad/blob/master/%2BburstDetection/generateSample.m):

- three K-means archetypes are fitted to real 100-pM SMAD trajectories and
  sampled according to their observed frequencies;
- cumulative Gaussian noise creates slow single-cell drift;
- one to sixteen candidate Gaussian bursts are sampled, with the original
  width-normalized separation rule rejecting ambiguous overlaps;
- the stimulus-driven initial response and accepted synthetic bursts receive
  dense labels.

The implementation also provides analytic SMAD-like archetypes so the generator
can be used without experimental inputs. As in MATLAB, the `height` parameter
multiplies a normalized Gaussian density and therefore controls burst area, not
its apex height. Arrays use the repository convention `(samples, 289)` and event
intervals use explicit half-open Python coordinates.

## Baselines and evaluation

Three neural architectures consume the normalized raw signal and its first
difference while retaining all 289 output positions. `BurstTCN` uses residual
blocks with exponentially increasing dilation for multi-scale context,
`BurstCNN` is a strictly local non-dilated comparator, and `BurstBiGRU` supplies
a bidirectional recurrent alternative. Weighted binary cross entropy addresses
the foreground/background imbalance. Each architecture is tuned in its own
persistent Optuna study before its best validation checkpoint is exported.

The alternative `02_unet_training_and_evaluation.ipynb` trains `BurstUNet`, a
1D encoder-decoder with pooling, linear upsampling, and concatenated skip
connections. Decoder stages target the exact encoder lengths, preserving all
289 output positions even though the sequence length is odd. It uses the same
train/validation protocol and a separate persistent U-Net Optuna study.

The classical comparator ports
[`stochastic-smad/+burstDetection/burstDetect.m`](https://github.com/lohex/stochastic-smad/blob/master/%2BburstDetection/burstDetect.m):
three Gaussian smoothing stages, trend removal and prominence-based peak
detection use the published default parameters. SciPy half-prominence widths
replace MATLAB `findpeaks` widths, so this is a transparent Python equivalent,
not a claim of bit-identical MATLAB output.

Model selection uses only validation loss and a validation-selected probability
threshold. The test split is opened once after these choices are frozen.
Reported metrics include:

- point precision, recall, F1, average precision and ROC AUC;
- event precision, recall and F1 after one-to-one interval matching by IoU;
- per-trajectory burst-count mean absolute error;
- center-time and duration mean absolute errors for matched events, in minutes.

An event is a contiguous positive interval after closing one-sample gaps and
removing events shorter than three samples. The default match criterion is
IoU >= 0.1. These choices are intentionally explicit and should be revisited
before applying the example to a new biological annotation protocol.

## Notebook workflow

| Notebook | Purpose |
| --- | --- |
| [`01_data_and_classical_baseline.ipynb`](notebooks/01_data_and_classical_baseline.ipynb) | Fit train-only SMAD archetypes, generate independent synthetic folds, inspect labels and evaluate the classical detector. |
| [`02_tcn_training_and_evaluation.ipynb`](notebooks/02_tcn_training_and_evaluation.ipynb) | Check TCN trainability, tune TCN/CNN/BiGRU variants, train their final candidates, and export validation-frozen checkpoints and thresholds without reading the hold-out results. |
| [`02_unet_training_and_evaluation.ipynb`](notebooks/02_unet_training_and_evaluation.ipynb) | Alternative 1D U-Net workflow with a memorization check, Optuna tuning, final training, validation event plots, and separate frozen artifacts. |
| [`03_holdout_test_and_model_comparison.ipynb`](notebooks/03_holdout_test_and_model_comparison.ipynb) | Load both frozen manifests, verify matching protocols, and compare TCN, CNN, BiGRU, U-Net, and the classical detector on the shared hold-out fold with metric bars and aligned probability/event panels. |

Reusable code lives in `src/burst_detection/`:

- `data.py`: archetypes, faithful synthetic generation, splits and preprocessing;
- `model.py`: length-preserving dilated TCN, local CNN, bidirectional GRU and 1D U-Net;
- `training.py`: deterministic training, positive weighting, early stopping,
  bounded inference, checkpointing and the overfit check;
- `tuning.py`: persistent per-architecture Optuna searches and trial analysis;
- `artifacts.py`: validation-selected checkpoint manifest for hold-out evaluation;
- `metrics.py`: probability cleanup, one-to-one event matching and metrics;
- `classical.py`: classical detector port;
- `visualization.py`: signal, truth, probability and predicted-event overlays.

## Run

From the repository root:

```bash
uv sync --extra torch --extra tuning
jupyter lab projects/burst_detection/notebooks/
```

The default teaching run generates 800/200/200 trajectories. The notebooks do
not copy synthetic arrays into the repository; seeds and generator parameters
fully define them. The final checkpoints and frozen hold-out manifest are
written to the ignored `notebooks/artifacts/` directory.

The U-Net alternative writes `best_unet.pt`, `burst_unet_optuna.db`, and
`unet_holdout_models.json` separately. Run both `02_` training notebooks before
Notebook 03, which loads both manifests automatically and includes all four neural
models in the comparison. The split definitions and selection protocols must match.
