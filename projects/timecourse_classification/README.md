# PyTorch time-course classification

## Goal

Classify the TGF-beta stimulation dose of single-cell trajectories while developing a reproducible workflow for temporal ML. The project separates controlled capacity/data scaling, architecture screening, final model-family comparison and explanation instead of treating model development as one undifferentiated hyperparameter search.

## ML engineering questions

- How much CNN capacity is useful for a given amount of training data?
- When is the baseline data-limited, capacity-limited or optimization-limited?
- Are first differences, frequency features, multiple temporal scales or long-range attention useful beyond simple CNN scaling?
- Can model families be compared without using the test fold for iterative decisions?
- Can attention weights and temporal attribution methods identify plausible regions of a time course?
- Which artifacts are required to reload and understand a trained experiment?

## Notebook workflow

| Notebook | Purpose | Engineering question |
| --- | --- | --- |
| [`01_data_visualization.ipynb`](notebooks/01_data_visualization.ipynb) | Train-only trajectories, statistical features and clustering | Which signal characteristics should guide modeling without leaking validation information? |
| [`02_cnn_training.ipynb`](notebooks/02_cnn_training.ipynb) | Memorization sanity check, nested data scaling, width scaling and compute-efficiency analysis | How should baseline CNN capacity scale with available training data, and where are data/capacity/optimization limits? |
| [`02_extended_cnns_training.ipynb`](notebooks/02_extended_cnns_training.ipynb) | Controlled screening, equal-budget architecture-specific Optuna searches and MLflow tracking | Which architecture/training configurations are performant, stable and computationally defensible? |
| [`02_attention_training.ipynb`](notebooks/02_attention_training.ipynb) | Positional self-attention, CNN attention pooling and hierarchical patch transformers | When is global context useful, and what can attention reveal about the decision? |
| [`03_model_family_comparison.ipynb`](notebooks/03_model_family_comparison.ipynb) | Final CNN, extended-CNN and attention comparison | Does validation-based selection transfer to test performance across accuracy, ranking and class-specific metrics? |
| [`04_explainable_ai.ipynb`](notebooks/04_explainable_ai.ipynb) | Grad-CAM, Integrated Gradients, temporal occlusion and deletion tests | Which temporal evidence is used by the top three Extended CNNs selected in Notebook 03? |

## Implementation

- `src/pytorch_timecourse_classification/models/` contains model definitions and the registry; the baseline CNN exposes a controlled width-scaling configuration.
- `data.py` loads/preprocesses fixed folds and provides synchronized fold subsetting.
- `training.py` provides architecture-independent optimization, schedules, class weighting, balanced sampling and early stopping.
- `experiments.py` keeps generic model creation, trainable-parameter counting, training orchestration and artifact verification in importable code.
- `scaling.py` defines nested stratified training subsets, the mini-dataset memorization check, the controlled `N x P x learning-rate` grid, empirical scaling summaries and compute-efficiency selection.
- `tuning.py` supports repeated-seed ablations and Optuna scans for broader architecture searches.
- `analysis.py` contains common metrics, history comparisons and attention overlays.
- `artifacts.py` saves portable checkpoints, training histories, selection manifests and strategy metadata.
- `tracking.py` connects architecture-specific Optuna trials to explicit local MLflow runs.
- `explainability.py` provides model-aware Grad-CAM and model-independent input attribution and perturbation diagnostics.

## Evaluation contract

The fixed folds in `Data/tgfb_stimulation_labels.csv` are loaded directly. Preprocessing is fitted on train only. The tune (`validate`) fold controls early stopping, the per-`N x P` learning-rate choice, architecture tuning, model selection, classification reports and confusion matrices. The model-development notebooks do not access test data; `03_model_family_comparison.ipynb` freezes family winners and the top three Extended CNNs before opening test. Notebook 04 consumes that tune-only selection manifest and does not rerank models.

Accuracy is complemented by class-wise precision and recall, confusion matrices, Average Precision, ROC and precision-recall curves. For scaling experiments, cross-entropy is the primary quantity because thresholded classification metrics can saturate while probabilistic fit continues to change. Training histories retain the selected epoch so convergence and generalization gaps can be inspected rather than reporting only a final score.

## Artifacts

Each deployable experiment stores a rebuildable checkpoint, train/tune history and `metadata.json` under `notebooks/artifacts/<experiment_name>/`. The CNN scaling notebook stores its grid tables below `notebooks/artifacts/cnn_scaling/` so expensive matrix runs can be reused without conflating them with deployable model artifacts.

Optuna/MLflow experiments are optional and live below `notebooks/artifacts/experiments/`, separate from deployable candidate folders. Install their dependencies together with the CUDA-enabled project environment using `uv sync --extra torch --extra experiment` (with the configured PyTorch CUDA index for the local platform).

## Run

```bash
uv sync --extra torch
jupyter lab projects/timecourse_classification/notebooks/
```

For Optuna scans, MLflow tracking and Captum attribution with uv:

```bash
uv sync --extra torch --extra experiment
```
