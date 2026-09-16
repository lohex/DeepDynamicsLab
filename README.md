# DeepLearningExamples

Small, self-contained PyTorch projects for biological time-course data. The repository develops the workflows from data inspection and fixed splits through reproducible training, tuning, interpretation, model comparison and generative AI. Notebooks document questions and results; reusable implementation lives in importable Python packages.

## Repository structure

```text
Data/                         Example time-course datasets
src/deeplearning_examples/    Shared data loading and preparation
projects/
  <project>/
    README.md                  Goal, method and execution notes
    notebooks/                Experiment orchestration and interpretation
    src/<package>/             Project-specific implementation
```

## Projects

| Project | Scientific question | ML engineering focus |
| --- | --- | --- |
| [Time-course classification](projects/timecourse_classification/README.md) | Which temporal representations distinguish TGF-beta stimulation doses? | Controlled ablations, optimization versus architecture, reproducible model selection and temporal explanations |
| [Burst detection](projects/burst_detection/README.md) | Where do bursts occur in single-cell SMAD trajectories? | Dense temporal prediction, imbalanced segmentation and point/event-level evaluation |
| [Generative AI: time-course generation](projects/timecourse-generation/README.md) | Can conditional generative models reproduce realistic, dose-specific single-cell trajectories? | GAN, Wasserstein/f-GAN and VAE training with held-out distribution and diversity diagnostics |

Each project README contains its notebook workflow and run instructions: [time-course classification](projects/timecourse_classification/README.md), [burst detection](projects/burst_detection/README.md) and [generative AI](projects/timecourse-generation/README.md).

## ML engineering questions

The projects are organized around questions that recur in real ML systems:

- **Data validity:** What is the independent experimental unit, and how are train, validation and test splits fixed before modeling to prevent leakage?
- **Training or model?** When performance changes, is the cause optimization, regularization and sampling, or the architecture and its inductive bias?
- **Reliability:** Is a result stable across seeds, does training converge, and do the histories indicate underfitting, overfitting or unstable optimization?
- **Search strategy:** Which hypotheses deserve controlled ablations, and when is a broader Optuna search more appropriate?
- **Evaluation discipline:** How are models selected without repeatedly consulting the test fold, and which metrics expose class-specific errors hidden by accuracy?
- **Interpretability:** Does a model use plausible temporal evidence, and do attention or attribution maps support that conclusion?
- **Reproducibility:** Which preprocessing state, model configuration, training strategy and checkpoint must be saved to rebuild an experiment?
- **Generative AI quality:** Do synthetic trajectories preserve dose-dependent distributions, dynamics and diversity rather than merely optimizing a discriminator or reconstruction loss?

Together, the projects move from trustworthy dataset construction through discriminative model development to conditional generative AI. The validation-selected classifier supports the generation project as an independent test of whether generated trajectories retain dose information.

## Design conventions

- Notebooks contain data selection, experiment configuration, plots and interpretation.
- Neural-network definitions live in each project's `model.py` or `models/` package.
- Optimization lives in `training.py` and is exposed as a function, not a trainer class.
- Shared experiment orchestration, artifact persistence and analysis live in project `src/` packages rather than notebook helper functions.
- Shared dataset imports and representations live in `src/deeplearning_examples/`.
- Fixed train, validation and test assignments are stored with the data; preprocessing is fitted on train only and test remains untouched until final evaluation.
- Rebuildable checkpoints are accompanied by training histories and strategy metadata.


## Installation

The repository uses `uv`. Install the PyTorch projects from the repository root with:

```bash
uv sync --extra torch
```

Add optional tuning and experiment-tracking dependencies as needed:

```bash
uv sync --extra torch --extra tuning
uv sync --extra torch --extra experiment
uv sync --all-extras
```

The equivalent editable pip installation for tracked tuning is `python -m pip install -e ".[torch,experiment]"`.

