import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from burst_detection.artifacts import (
    load_holdout_manifest,
    preprocessor_from_checkpoint,
    save_holdout_manifest,
)
from burst_detection.classical import detect_classical_bursts
from burst_detection.data import (
    BurstEvent,
    SyntheticBurstConfig,
    default_smad_archetypes,
    fit_preprocessor,
    generate_dataset_splits,
    generate_synthetic_bursts,
    mask_to_events,
    targets_to_tensor,
)
from burst_detection.metrics import (
    evaluate_burst_predictions,
    match_events,
    probabilities_to_mask,
    select_threshold,
)
from burst_detection.model import BurstBiGRU, BurstCNN, BurstTCN, create_burst_model
from burst_detection.training import (
    TrainingConfig,
    load_checkpoint,
    predict_probabilities,
    train_model,
)


class SyntheticBurstDataTests(unittest.TestCase):
    def test_generator_is_reproducible_and_aligned(self) -> None:
        first = generate_synthetic_bursts(12, random_seed=17)
        second = generate_synthetic_bursts(12, random_seed=17)
        np.testing.assert_array_equal(first.signals, second.signals)
        np.testing.assert_array_equal(first.labels, second.labels)
        self.assertEqual(first.signals.shape, (12, 289))
        self.assertEqual(first.labels.shape, first.instance_labels.shape)
        self.assertTrue(first.labels[:, 6:40].all())
        for labels, events in zip(first.labels, first.events):
            self.assertEqual(events, tuple(mask_to_events(labels)))

    def test_splits_use_independent_seed_streams(self) -> None:
        splits = generate_dataset_splits(
            train_size=4, validation_size=4, test_size=4, random_seed=11
        )
        self.assertFalse(np.array_equal(splits.train.signals, splits.validation.signals))
        repeated = generate_dataset_splits(
            train_size=4, validation_size=4, test_size=4, random_seed=11
        )
        np.testing.assert_array_equal(splits.test.signals, repeated.test.signals)

    def test_preprocessing_has_raw_and_derivative_channels(self) -> None:
        signals = generate_synthetic_bursts(6, random_seed=2).signals
        preprocessor = fit_preprocessor(signals)
        features = preprocessor.transform(signals)
        self.assertEqual(tuple(features.shape), (6, 2, 289))
        self.assertAlmostEqual(float(features[:, 0].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(features[:, 0].std(unbiased=False)), 1.0, places=5)
        expected_first_derivative = torch.full(
            (6,), -preprocessor.derivative_mean / preprocessor.derivative_std
        )
        self.assertTrue(
            torch.allclose(features[:, 1, 0], expected_first_derivative)
        )

    def test_custom_length_is_supported(self) -> None:
        config = SyntheticBurstConfig(
            length=65,
            center_min=20,
            center_max=55,
            initial_label_stop=15,
            maximum_peak_attempts=3,
        )
        dataset = generate_synthetic_bursts(
            3,
            archetypes=default_smad_archetypes(65),
            config=config,
            random_seed=4,
        )
        self.assertEqual(dataset.signals.shape, (3, 65))


class BurstMetricTests(unittest.TestCase):
    def test_gap_closing_and_short_event_removal(self) -> None:
        probabilities = np.asarray([[0, 1, 1, 0, 1, 1, 0, 1, 0]], dtype=float)
        cleaned = probabilities_to_mask(
            probabilities, threshold=0.5, minimum_duration=3, bridge_gap=1
        )
        np.testing.assert_array_equal(
            cleaned,
            np.asarray([[0, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=bool),
        )

    def test_event_metrics_use_one_to_one_matching(self) -> None:
        truth = np.asarray([[0, 1, 1, 1, 0, 0, 1, 1, 0]], dtype=bool)
        predicted = np.asarray([[0, 0, 1, 1, 1, 0, 1, 1, 0]], dtype=float)
        metrics = evaluate_burst_predictions(
            truth,
            predicted,
            threshold=0.5,
            minimum_duration=1,
            bridge_gap=0,
            minimum_iou=0.1,
            sample_interval_minutes=5,
        )
        self.assertEqual(metrics.matched_event_count, 2)
        self.assertEqual(metrics.event_f1, 1.0)
        self.assertEqual(metrics.burst_count_mae, 0.0)
        self.assertEqual(metrics.center_time_mae_minutes, 2.5)
        self.assertEqual(metrics.duration_mae_minutes, 0.0)

    def test_assignment_cannot_reuse_one_prediction(self) -> None:
        truth = [BurstEvent(0, 4), BurstEvent(5, 9)]
        predicted = [BurstEvent(0, 9)]
        self.assertEqual(len(match_events(truth, predicted, minimum_iou=0.1)), 1)

    def test_threshold_is_selected_from_candidates(self) -> None:
        truth = np.asarray([[0, 1, 1, 0]], dtype=bool)
        scores = np.asarray([[0.1, 0.7, 0.8, 0.4]])
        selection = select_threshold(
            truth,
            scores,
            thresholds=np.asarray([0.3, 0.6, 0.9]),
            minimum_duration=1,
            bridge_gap=0,
        )
        self.assertEqual(selection.threshold, 0.6)


class BurstModelAndTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        config = SyntheticBurstConfig(
            length=65,
            center_min=20,
            center_max=55,
            initial_label_stop=15,
            maximum_peak_attempts=3,
        )
        dataset = generate_synthetic_bursts(
            12,
            archetypes=default_smad_archetypes(65),
            config=config,
            random_seed=8,
        )
        self.preprocessor = fit_preprocessor(dataset.signals)
        self.features = self.preprocessor.transform(dataset.signals)
        self.targets = targets_to_tensor(dataset.labels)

    def test_tcn_preserves_temporal_resolution(self) -> None:
        model = BurstTCN(
            input_channels=2,
            hidden_channels=8,
            dilations=(1, 2, 4),
            dropout=0.0,
        )
        output = model(self.features[:3])
        self.assertEqual(tuple(output.shape), (3, 65))
        self.assertEqual(model.receptive_field, 57)

    def test_alternative_architectures_preserve_temporal_resolution(self) -> None:
        models = (
            BurstCNN(
                input_channels=2, hidden_channels=8, kernel_size=5,
                depth=3, dropout=0.0,
            ),
            BurstBiGRU(
                input_channels=2, hidden_size=8, num_layers=1, dropout=0.0,
            ),
        )
        for model in models:
            with self.subTest(model=model.model_name):
                output = model(self.features[:3])
                self.assertEqual(tuple(output.shape), (3, 65))
                rebuilt = create_burst_model(model.model_name, model.model_config)
                self.assertEqual(type(rebuilt), type(model))

    def test_training_writes_rebuildable_best_checkpoint(self) -> None:
        model = BurstTCN(
            input_channels=2,
            hidden_channels=4,
            dilations=(1, 2),
            dropout=0.0,
        )
        config = TrainingConfig(
            epochs=2,
            batch_size=4,
            patience=2,
            report_every=2,
            random_seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            history = train_model(
                model,
                self.features[:8],
                self.targets[:8],
                self.features[8:],
                self.targets[8:],
                config=config,
                device="cpu",
                checkpoint_path=path,
                preprocessor=self.preprocessor,
            )
            restored, metadata = load_checkpoint(path)
            probabilities = predict_probabilities(restored, self.features[8:], device="cpu")
        self.assertGreaterEqual(history.best_epoch, 1)
        self.assertEqual(probabilities.shape, (4, 65))
        self.assertEqual(metadata["model_name"], "burst_tcn")
        self.assertTrue(metadata["preprocessor"]["include_derivative"])

    def test_alternative_checkpoint_is_rebuildable(self) -> None:
        model = BurstCNN(
            input_channels=2, hidden_channels=4, kernel_size=3,
            depth=2, dropout=0.0,
        )
        config = TrainingConfig(
            epochs=1, batch_size=4, patience=1, report_every=1, random_seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_cnn.pt"
            train_model(
                model, self.features[:8], self.targets[:8],
                self.features[8:], self.targets[8:], config=config,
                device="cpu", checkpoint_path=path, preprocessor=self.preprocessor,
            )
            restored, metadata = load_checkpoint(path)
        self.assertIsInstance(restored, BurstCNN)
        self.assertEqual(metadata["model_name"], "burst_cnn")
        self.assertEqual(preprocessor_from_checkpoint(metadata), self.preprocessor)

    def test_classical_detector_returns_dense_mask(self) -> None:
        detected = detect_classical_bursts(self.features[:2, 0].numpy())
        self.assertEqual(detected.shape, (2, 65))
        self.assertEqual(detected.dtype, np.bool_)


class BurstNotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_json(self) -> None:
        notebook_dir = Path("projects/burst_detection/notebooks")
        for path in notebook_dir.glob("*.ipynb"):
            with path.open(encoding="utf-8") as handle:
                notebook = json.load(handle)
            self.assertEqual(notebook["nbformat"], 4)
            self.assertGreater(len(notebook["cells"]), 1)

    def test_holdout_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdout_models.json"
            save_holdout_manifest(
                path,
                candidates=[{
                    "architecture": "tcn",
                    "checkpoint": "best_tcn.pt",
                    "threshold": 0.4,
                    "validation_metrics": {"event_f1": 0.9},
                }],
                split_config={"train_size": 8, "validation_size": 4, "test_size": 4},
            )
            manifest = load_holdout_manifest(path)
        self.assertEqual(manifest["selection_fold"], "validation")
        self.assertEqual(manifest["candidates"][0]["checkpoint"], "best_tcn.pt")


if __name__ == "__main__":
    unittest.main()
