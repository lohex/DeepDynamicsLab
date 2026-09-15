import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from pytorch_timecourse_classification.analysis import predict_classes
from pytorch_timecourse_classification.artifacts import (
    load_model,
    load_selection_manifest,
    save_experiment_artifacts,
    save_selection_manifest,
)
from pytorch_timecourse_classification.data import (
    FoldData,
    Preprocessor,
    preprocessors_compatible,
)
from pytorch_timecourse_classification.models.cnn import CNNClassifier
from pytorch_timecourse_classification.models.attention import AttentionClassifier
from pytorch_timecourse_classification.models.cnn_attention_pooling import (
    CNNAttentionPoolingClassifier,
)
from pytorch_timecourse_classification.models.extended_cnns import ExtendedCNNClassifier
from pytorch_timecourse_classification.models.hierarchical_patch_transformer import (
    HierarchicalPatchTransformerClassifier,
)
from pytorch_timecourse_classification.explainability import (
    explainable_layers,
    integrated_gradients,
    layer_gradcam,
    temporal_window_deletion,
)
from pytorch_timecourse_classification.tracking import MLflowTrackingConfig
from pytorch_timecourse_classification.training import TrainingConfig, train
from pytorch_timecourse_classification.tuning import (
    AblationSpec,
    add_paired_reference_deltas,
    run_ablations,
    run_optuna_scan,
    summarize_ablations,
    suggest_attention_model_config,
    suggest_attention_training_config,
    suggest_extended_cnn_model_config,
    OptunaScanConfig,
)


class DeterministicTrial:
    """Small Optuna-like object for validating conditional search spaces."""

    def __init__(self, architecture: str) -> None:
        self.architecture = architecture
        self.suggested_names: list[str] = []

    def suggest_categorical(self, name: str, choices: list | tuple):
        self.suggested_names.append(name)
        if name == "model.architecture":
            return self.architecture
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
        self.suggested_names.append(name)
        return low

    def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
        self.suggested_names.append(name)
        return low


def synthetic_fold(seed: int, sample_count: int = 12) -> FoldData:
    rng = np.random.default_rng(seed)
    targets = np.arange(sample_count) % 2
    trajectories = rng.normal(size=(sample_count, 16)).astype(np.float32)
    trajectories += targets[:, None] * 0.8
    return FoldData(
        trajectories=trajectories,
        doses=np.where(targets == 0, "0pM", "1pM"),
        years=np.full(sample_count, 2020),
        features=torch.from_numpy(trajectories[:, None, :]),
        targets=torch.tensor(targets, dtype=torch.long),
    )


class NaNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.flatten(1)) * float("nan")


class TimecourseTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_fold = synthetic_fold(1)
        self.tune_fold = synthetic_fold(2, sample_count=8)
        self.preprocessor = Preprocessor(("0pM", "1pM"), 0.0, 1.0)
        self.config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            scheduler_strategy="none",
            patience=2,
            report_every=2,
            class_balance_strategy="none",
            gradient_clip_norm=1.0,
        )
        self.model_config = {
            "channels": (2,),
            "kernel_sizes": (3,),
            "dilations": (1,),
            "pool_size": 2,
            "hidden_dim": 2,
            "dropout": 0.0,
            "residual": False,
        }

    def test_training_history_contains_complete_epoch_series(self) -> None:
        model = CNNClassifier(input_length=16, num_classes=2, **self.model_config)
        history = train(
            model,
            self.train_fold.features,
            self.train_fold.targets,
            self.tune_fold.features,
            self.tune_fold.targets,
            config=self.config,
            device="cpu",
        )
        for field in (
            "training_loss",
            "validation_loss",
            "training_accuracy",
            "validation_accuracy",
            "training_macro_f1",
            "validation_macro_f1",
            "learning_rate",
        ):
            values = np.asarray(getattr(history, field))
            self.assertEqual(len(values), 2)
            self.assertTrue(np.isfinite(values).all())

        with tempfile.TemporaryDirectory() as directory:
            paths = save_experiment_artifacts(
                Path(directory),
                model=model,
                history=history,
                preprocessor=self.preprocessor,
                training_targets=self.train_fold.targets,
                validation_targets=self.tune_fold.targets,
                device="cpu",
            )
            with np.load(paths.history) as saved_history:
                self.assertIn("training_macro_f1", saved_history)
                self.assertIn("validation_macro_f1", saved_history)
                self.assertIn("learning_rate", saved_history)
            restored, _ = load_model(paths.model, device="cpu")
            model.eval()
            with torch.no_grad():
                expected_logits = model(self.tune_fold.features)
                restored_logits = restored(self.tune_fold.features)
            torch.testing.assert_close(
                restored_logits, expected_logits, rtol=0.0, atol=0.0
            )

    def test_predict_classes_uses_bounded_batches(self) -> None:
        model = AttentionClassifier(
            input_length=16,
            num_classes=2,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            dropout=0.0,
            pooling="mean",
            norm_first=True,
        )
        model.eval()
        observed_batch_sizes = []

        def record_batch(module, arguments):
            observed_batch_sizes.append(len(arguments[0]))

        handle = model.register_forward_pre_hook(record_batch)
        try:
            predictions = predict_classes(
                model, self.train_fold, device="cpu", batch_size=5
            )
        finally:
            handle.remove()
        self.assertEqual(tuple(predictions.shape), (len(self.train_fold.features),))
        self.assertEqual(observed_batch_sizes, [5, 5, 2])

        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            predict_classes(model, self.train_fold, device="cpu", batch_size=0)

    def test_preprocessor_compatibility_tolerates_serialization_noise(self) -> None:
        reference = Preprocessor(("0pM", "1pM"), 0.5543460792941041, 0.33032176022350446)
        rounded = Preprocessor(("0pM", "1pM"), 0.5543460792941042, 0.3303217602235046)
        changed_scale = Preprocessor(("0pM", "1pM"), 0.5543460792941041, 0.34)
        changed_order = Preprocessor(("1pM", "0pM"), 0.5543460792941041, 0.33032176022350446)

        self.assertTrue(preprocessors_compatible(reference, rounded))
        self.assertFalse(preprocessors_compatible(reference, changed_scale))
        self.assertFalse(preprocessors_compatible(reference, changed_order))

    def test_non_finite_metrics_raise_explicit_error(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            train(
                NaNClassifier(),
                self.train_fold.features,
                self.train_fold.targets,
                self.tune_fold.features,
                self.tune_fold.targets,
                config=self.config,
                device="cpu",
            )

    def test_paired_deltas_and_summary(self) -> None:
        frame = pd.DataFrame(
            {
                "name": ["reference", "candidate", "reference", "candidate"],
                "random_seed": [42, 42, 43, 43],
                "tune_loss": [1.0, 0.9, 1.1, 1.0],
                "tune_accuracy": [0.5, 0.6, 0.4, 0.5],
                "tune_macro_f1": [0.40, 0.45, 0.50, 0.60],
            }
        )
        paired = add_paired_reference_deltas(frame, reference_name="reference")
        candidate_deltas = paired.loc[
            paired["name"] == "candidate", "tune_macro_f1_delta"
        ].to_numpy()
        np.testing.assert_allclose(candidate_deltas, [0.05, 0.10])
        summary = summarize_ablations(paired)
        self.assertAlmostEqual(summary.loc["candidate", "macro_f1_delta_mean"], 0.075)

    def test_run_ablations_cpu_smoke(self) -> None:
        specs = (
            AblationSpec(
                name="reference",
                model_config=self.model_config,
                training_config=self.config,
                random_seeds=(42,),
            ),
        )
        results, frame = run_ablations(
            specs,
            model_type=CNNClassifier,
            training_fold=self.train_fold,
            tuning_fold=self.tune_fold,
            preprocessor=self.preprocessor,
            device="cpu",
        )
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].failed)
        self.assertEqual(int(frame.loc[0, "epochs_run"]), 2)
        self.assertTrue(np.isfinite(frame.loc[0, "duration_seconds"]))

    def test_extended_cnn_search_space_builds_every_architecture(self) -> None:
        architectures = ExtendedCNNClassifier.supported_architectures
        inputs = torch.randn(4, 1, 16)
        for architecture in architectures:
            trial = DeterministicTrial(architecture)
            config = suggest_extended_cnn_model_config(
                trial,
                {},
                architectures=(architecture,),
            )
            model = ExtendedCNNClassifier(
                input_length=16,
                num_classes=2,
                **config,
            )
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (4, 2))
            layers = explainable_layers(model)
            self.assertTrue(layers)
            self.assertTrue(all(layer.domain in {"time", "frequency"} for layer in layers))
            if architecture == "multi_scale":
                self.assertIn("model.output_channels", trial.suggested_names)
                self.assertNotIn("model.adaptive_pool_size", trial.suggested_names)
                self.assertNotIn("model.depth", trial.suggested_names)
            elif architecture == "tcn":
                self.assertIn("model.tcn_blocks", trial.suggested_names)
                self.assertNotIn("model.adaptive_pool_size", trial.suggested_names)
            elif architecture == "shift_robust":
                self.assertIn("model.shift_depth", trial.suggested_names)
                self.assertNotIn("model.dilation_profile", trial.suggested_names)
                self.assertNotIn("model.adaptive_pool_size", trial.suggested_names)
            else:
                self.assertIn("model.adaptive_pool_size", trial.suggested_names)

    def test_attention_search_space_builds_every_architecture(self) -> None:
        model_types = {
            "mean_attention": AttentionClassifier,
            "cls_attention": AttentionClassifier,
            "cnn_attention_pooling": CNNAttentionPoolingClassifier,
            "hierarchical_patch": HierarchicalPatchTransformerClassifier,
        }
        inputs = torch.randn(4, 1, 16)
        for architecture, model_type in model_types.items():
            trial = DeterministicTrial(architecture)
            config = suggest_attention_model_config(
                trial, {}, architecture=architecture
            )
            model = model_type(input_length=16, num_classes=2, **config)
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (4, 2))
            self.assertIn("model.architecture", trial.suggested_names)

    def test_attention_training_space_uses_memory_conscious_batches(self) -> None:
        trial = DeterministicTrial("mean_attention")
        config = suggest_attention_training_config(trial, self.config)
        self.assertIn(config.batch_size, {16, 32, 64, 128})
        self.assertNotIn(config.batch_size, {256, 512})

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna extra not installed")
    def test_total_optuna_budget_is_not_added_twice(self) -> None:
        model_config = {
            "architecture": "first_difference",
            "branch_channels": (2, 4),
            "kernel_sizes": (3, 3),
            "dilations": (1, 1),
            "hidden_dims": (4,),
            "adaptive_pool_size": 1,
            "dropout": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            config = OptunaScanConfig(
                scope="model",
                n_trials=1,
                n_trials_mode="total",
                study_name="fixed-budget-smoke",
                storage=f"sqlite:///{Path(directory) / 'study.db'}",
                n_startup_trials=0,
                pruning_warmup_epochs=1,
            )
            arguments = dict(
                model_type=ExtendedCNNClassifier,
                base_model_config=model_config,
                base_training_config=self.config,
                training_fold=self.train_fold,
                tuning_fold=self.tune_fold,
                preprocessor=self.preprocessor,
                device="cpu",
                scan_config=config,
                model_config_suggester=lambda trial, base: dict(base),
            )
            first = run_optuna_scan(**arguments)
            second = run_optuna_scan(**arguments)
            self.assertEqual(len(first.trials), 1)
            self.assertEqual(len(second.trials), 1)

    def test_temporal_window_deletion_returns_matched_controls(self) -> None:
        model = CNNClassifier(input_length=16, num_classes=2, **self.model_config)
        inputs = self.tune_fold.features[:4]
        targets = self.tune_fold.targets[:4]
        attributions = torch.rand_like(inputs)
        deletion = temporal_window_deletion(
            model,
            inputs,
            targets,
            attributions,
            window_size=3,
        )
        for key in ("top", "random", "top_start", "random_start"):
            self.assertEqual(tuple(deletion[key].shape), (4,))
        self.assertTrue(torch.isfinite(deletion["top"]).all())

    def test_selection_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "pytorch_timecourse_classification.artifacts.artifact_root",
                return_value=Path(directory),
            ):
                path = save_selection_manifest(
                    "extended_cnns_top3.json",
                    candidates=[
                        {
                            "rank": np.int64(1),
                            "experiment": "extended_cnn_tcn",
                            "macro_f1": np.float64(0.6),
                        }
                    ],
                    selection_fold="validate",
                    selection_metric="macro_f1",
                    source_notebook="03_model_family_comparison.ipynb",
                )
                self.assertTrue(path.is_file())
                restored = load_selection_manifest("extended_cnns_top3.json")
        self.assertEqual(restored["candidates"][0]["rank"], 1)
        self.assertEqual(restored["selection_fold"], "validate")

    @unittest.skipUnless(importlib.util.find_spec("captum"), "Captum extra not installed")
    def test_captum_attributions_have_input_resolution(self) -> None:
        model = ExtendedCNNClassifier(
            input_length=16,
            num_classes=2,
            architecture="first_difference",
            branch_channels=(2, 4),
            kernel_sizes=(3, 3),
            dilations=(1, 1),
            hidden_dims=(4,),
            adaptive_pool_size=1,
            dropout=0.0,
        )
        model.eval()
        inputs = self.tune_fold.features[:4]
        targets = self.tune_fold.targets[:4]
        integrated_batch_sizes = []

        def record_integrated_batch(module, arguments):
            integrated_batch_sizes.append(len(arguments[0]))

        handle = model.register_forward_pre_hook(record_integrated_batch)
        try:
            integrated = integrated_gradients(
                model,
                inputs,
                targets,
                steps=4,
                internal_batch_size=len(inputs),
            )
        finally:
            handle.remove()
        gradcam = layer_gradcam(
            model,
            inputs,
            targets,
            explainable_layers(model)[0],
            output_length=16,
        )
        self.assertEqual(tuple(integrated.shape), tuple(inputs.shape))
        self.assertEqual(tuple(gradcam.shape), tuple(inputs.shape))
        self.assertTrue(torch.isfinite(integrated).all())
        self.assertTrue(torch.isfinite(gradcam).all())
        self.assertLessEqual(max(integrated_batch_sizes), len(inputs))
        with self.assertRaisesRegex(
            ValueError, "at least the number of input samples"
        ):
            integrated_gradients(
                model,
                inputs,
                targets,
                steps=4,
                internal_batch_size=len(inputs) - 1,
            )

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "MLflow extra not installed")
    def test_optuna_trial_is_logged_to_mlflow(self) -> None:
        model_config = {
            "architecture": "first_difference",
            "branch_channels": (2, 4),
            "kernel_sizes": (3, 3),
            "dilations": (1, 1),
            "hidden_dims": (4,),
            "adaptive_pool_size": 1,
            "dropout": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            tracking = MLflowTrackingConfig.local(
                Path(directory) / "mlflow",
                experiment_name="smoke",
                parent_run_name="study",
            )
            study = run_optuna_scan(
                model_type=ExtendedCNNClassifier,
                base_model_config=model_config,
                base_training_config=self.config,
                training_fold=self.train_fold,
                tuning_fold=self.tune_fold,
                preprocessor=self.preprocessor,
                device="cpu",
                scan_config=OptunaScanConfig(
                    scope="model",
                    n_trials=1,
                    n_startup_trials=0,
                    pruning_warmup_epochs=1,
                ),
                model_config_suggester=lambda trial, base: dict(base),
                mlflow_tracking=tracking,
            )
            self.assertEqual(len(study.trials), 1)
            self.assertTrue((Path(directory) / "mlflow" / "mlflow.db").is_file())


if __name__ == "__main__":
    unittest.main()
