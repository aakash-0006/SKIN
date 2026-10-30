"""Unit tests for online image augmentation, class weighting safeguards, and split integrity."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset import CLASSES, PROJECT_DIR, resolve_split_paths
from src.preprocessing import build_augmentation_layer, apply_augmentation_to_model
from src.train import compute_safe_class_weights, _check_output_protection, _require_split_integrity, PROTECTED_PATHS


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.aug = build_augmentation_layer(
            rotation_degrees=20.0,
            zoom_pct=0.08,
            translation_pct=0.05,
            name="test_aug",
        )

    def test_sublayers_and_config(self):
        layer_names = [l.name for l in self.aug.layers]
        self.assertIn("aug_hflip", layer_names)
        self.assertIn("aug_rotation", layer_names)
        self.assertIn("aug_zoom", layer_names)
        self.assertIn("aug_translation", layer_names)

        flip = self.aug.get_layer("aug_hflip")
        self.assertEqual(flip.mode, "horizontal")  # Vertical flip MUST NOT be enabled

        rot = self.aug.get_layer("aug_rotation")
        # 20 deg / 360 deg ≈ 0.05555
        self.assertAlmostEqual(rot.factor[1], 20.0 / 360.0, places=3)

    def test_inactive_during_evaluation(self):
        # In eval (training=False), output must be identical to input
        x = np.arange(224 * 224 * 3, dtype=np.float32).reshape(1, 224, 224, 3) / 1000.0
        out_eval = self.aug(x, training=False).numpy()
        np.testing.assert_array_equal(x, out_eval)

    def test_active_during_training(self):
        # In training (training=True), output should differ from input due to transformations
        x = np.arange(224 * 224 * 3, dtype=np.float32).reshape(1, 224, 224, 3) / 1000.0
        diffs = []
        for _ in range(5):
            out_train = self.aug(x, training=True).numpy()
            diffs.append(np.max(np.abs(x - out_train)))
        self.assertTrue(any(d > 0 for d in diffs), "Augmentation produced no variance across 5 stochastic trials")


class TestClassWeights(unittest.TestCase):
    def test_weights_calculation_and_cap(self):
        # Simulating training counts: [akiec: 690, bcc: 2396, bkl: 1746, df: 173, mel: 3138, nv: 8229, vasc: 174]
        counts = [690, 2396, 1746, 173, 3138, 8229, 174]
        y_train = np.repeat(np.arange(7), counts)

        cw, log_info = compute_safe_class_weights(y_train, classes=CLASSES, strategy="balanced_capped", max_cap=5.0)

        # Check that weights dictionary covers all 7 classes
        self.assertEqual(len(cw), 7)
        self.assertEqual(set(cw.keys()), set(range(7)))

        # Rare classes df (idx 3) and vasc (idx 6) raw weight is ~13.6, should be capped at 5.0
        self.assertEqual(cw[3], 5.0)
        self.assertEqual(cw[6], 5.0)

        # Majority class nv (idx 5) raw weight is ~0.287, should not be capped
        self.assertAlmostEqual(cw[5], 0.2872, places=3)

        # Check that log_info records the counts correctly
        self.assertEqual(log_info["total_training_samples"], sum(counts))
        self.assertEqual(log_info["class_counts"]["df"], 173)
        self.assertEqual(log_info["class_counts"]["nv"], 8229)


class TestSplitIntegrityAndProtection(unittest.TestCase):
    def test_protected_files(self):
        # Attempting to target protected paths must raise ValueError
        for protected in PROTECTED_PATHS:
            with self.assertRaises(ValueError):
                _check_output_protection(protected)

        # Allowed new output paths should not raise
        _check_output_protection("models/isic_augmented_v1.keras")
        _check_output_protection("outputs/augmentation_training_history.csv")

    def test_frozen_test_manifest(self):
        splits_path = PROJECT_DIR / "outputs" / "splits.csv"
        combined_path = PROJECT_DIR / "outputs" / "combined_splits.csv"

        self.assertTrue(splits_path.is_file(), "outputs/splits.csv is missing")
        self.assertTrue(combined_path.is_file(), "outputs/combined_splits.csv is missing")

        s1 = pd.read_csv(splits_path)
        s2 = pd.read_csv(combined_path)

        t1 = s1.loc[s1.split == "test", "image_id"].astype(str).tolist()
        t2 = s2.loc[s2.split == "test", "image"].astype(str).tolist()

        self.assertEqual(len(t1), 1494)
        self.assertEqual(t1, t2, "Frozen HAM test partition in combined_splits.csv does not match splits.csv!")

        # Verify no lesion leakage in combined_splits.csv
        _require_split_integrity(s2)


class TestModelReconstruction(unittest.TestCase):
    def test_model_preserves_weights_and_eval_output(self):
        init_model_path = PROJECT_DIR / "models" / "isic_finetuned.keras"
        if not init_model_path.is_file():
            init_model_path = PROJECT_DIR / "models" / "best_model.keras"

        model = tf.keras.models.load_model(init_model_path, compile=False)
        rebuilt = apply_augmentation_to_model(model)

        self.assertEqual(rebuilt.input_shape, (None, 224, 224, 3))
        self.assertEqual(rebuilt.output_shape, (None, 7))

        # Check that eval predictions are 100% identical between source and rebuilt model
        dummy = np.random.uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
        pred_orig = model(dummy, training=False).numpy()
        pred_rebuilt = rebuilt(dummy, training=False).numpy()
        np.testing.assert_allclose(pred_orig, pred_rebuilt, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
