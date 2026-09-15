"""U-Net temporal alignment, training, and persisted search integration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from burst_detection import BurstUNet, create_burst_model
from burst_detection.artifacts import preprocessor_from_checkpoint
from burst_detection.data import fit_preprocessor, generate_synthetic_bursts, targets_to_tensor
from burst_detection.training import TrainingConfig, load_checkpoint, predict_probabilities, train_model
from burst_detection.tuning import BurstOptunaConfig, best_burst_model_configs, run_burst_model_search


class BurstUNetTests(unittest.TestCase):
    def test_odd_even_and_minimum_lengths_with_backward(self):
        for depth in (2, 3, 4):
            model = BurstUNet(base_channels=2, depth=depth, dropout=0.0)
            for length in (2**depth, 65, 289, 300):
                with self.subTest(depth=depth, length=length):
                    model.zero_grad(set_to_none=True)
                    features = torch.randn(1, 2, length, requires_grad=True)
                    logits = model(features)
                    self.assertEqual(logits.shape, (1, length))
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, torch.rand_like(logits),
                    ).backward()
                    self.assertTrue(torch.isfinite(features.grad).all())
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_configurations_and_input_shapes(self):
        for config in ({'base_channels': 0}, {'depth': 0}, {'kernel_size': 4},
                       {'dropout': 1.0}, {'input_channels': 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                BurstUNet(**config)
        model = BurstUNet(depth=3)
        for shape in ((2, 289), (2, 3, 289), (2, 2, 7)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))

    def test_training_checkpoint_reproduces_predictions(self):
        data = generate_synthetic_bursts(12, random_seed=18)
        preprocessor = fit_preprocessor(data.signals[:8])
        features = preprocessor.transform(data.signals)
        targets = targets_to_tensor(data.labels)
        model = BurstUNet(base_channels=4, depth=2, kernel_size=3, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'best_unet.pt'
            history = train_model(
                model, features[:8], targets[:8], features[8:], targets[8:],
                config=TrainingConfig(epochs=2, batch_size=4, patience=2),
                device='cpu', checkpoint_path=checkpoint_path, preprocessor=preprocessor,
            )
            restored, metadata = load_checkpoint(checkpoint_path, device='cpu')
            expected = predict_probabilities(model, features[8:], device='cpu')
            actual = predict_probabilities(restored, features[8:], device='cpu')
        self.assertIsInstance(restored, BurstUNet)
        self.assertEqual(metadata['model_config'], model.model_config)
        self.assertEqual(preprocessor_from_checkpoint(metadata), preprocessor)
        self.assertGreaterEqual(history.best_epoch, 1)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(actual.shape, (4, 289))
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_persistent_unet_search_can_resume_and_rebuild_best_model(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest('Optuna optional dependency is not installed.')
        features = torch.randn(8, 2, 33)
        targets = (features[:, 0] > 0).float()
        with tempfile.TemporaryDirectory() as directory:
            config = BurstOptunaConfig(
                n_trials=1, study_name='unet-test',
                storage=f'sqlite:///{Path(directory) / "study.db"}',
            )
            kwargs = dict(
                base_training_config=TrainingConfig(epochs=1, patience=1),
                config=config, device='cpu',
            )
            study = run_burst_model_search(
                'unet', features[:4], targets[:4], features[4:], targets[4:], **kwargs,
            )
            resumed = run_burst_model_search(
                'unet', features[:4], targets[:4], features[4:], targets[4:], **kwargs,
            )
            self.assertEqual(len(resumed.trials), 1)
            self.assertEqual(study.best_value, resumed.best_value)
            name, model_config, training_config = best_burst_model_configs(
                resumed, architecture='unet', final_epochs=2, final_patience=1,
            )
        self.assertEqual(name, 'burst_unet')
        rebuilt = create_burst_model(name, model_config)
        self.assertEqual(rebuilt(features).shape, targets.shape)
        self.assertEqual(training_config.epochs, 2)
        self.assertEqual(training_config.patience, 1)


if __name__ == '__main__':
    unittest.main()
