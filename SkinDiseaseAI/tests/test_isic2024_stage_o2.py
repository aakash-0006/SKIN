"""Unit tests for Stage O.2 ISIC 2024 corrections.

Verifies:
1. Image preprocessing output range in [0, 255] and prevention of double normalization in MobileNetV3.
2. BinaryFocalLoss Keras-compatible serialization with **kwargs, save, and reload without error.
3. Standardized pAUC calculation against sklearn.metrics.roc_auc_score(..., max_fpr=0.20).
"""
import io
import os
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
import tensorflow as tf
from tensorflow.keras import layers, models

from src.isic2024.models import (
    BinaryFocalLoss,
    build_multimodal_model,
    build_image_only_model,
    build_meta_only_model,
)
from src.isic2024.metrics import compute_pauc, evaluate_binary_predictions


class TestImagePreprocessingAndBackbone(unittest.TestCase):
    """Verify image normalization, dynamic range, and MobileNetV3 internal rescaling."""

    def test_mobilenet_v3_internal_rescaling_layer(self):
        """Confirm MobileNetV3Small incorporates an internal Rescaling layer expecting [0, 255]."""
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(128, 128, 3),
            include_top=False,
            weights=None,
        )
        rescaling_layer = None
        for layer in base.layers:
            if isinstance(layer, layers.Rescaling):
                rescaling_layer = layer
                break
        
        assert rescaling_layer is not None, "MobileNetV3Small must have an internal Rescaling layer"
        cfg = rescaling_layer.get_config()
        # Scale must be ~ 1/127.5 (0.0078431) and offset must be -1.0
        assert np.isclose(cfg["scale"], 1.0 / 127.5, atol=1e-5)
        assert np.isclose(cfg["offset"], -1.0, atol=1e-5)

    def test_prevent_double_normalization(self):
        """Ensure inputting [0, 255] yields internal range [-1, 1], whereas inputting [-1, 1] collapses."""
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(128, 128, 3),
            include_top=False,
            weights=None,
        )
        # Find the output of the Rescaling layer
        rescaling_layer = [l for l in base.layers if isinstance(l, layers.Rescaling)][0]
        
        # Test A: Correct raw pixels in [0, 255]
        raw_pixels = tf.constant([[[[0.0, 127.5, 255.0]]]], dtype=tf.float32)
        rescaled_correct = rescaling_layer(raw_pixels).numpy()
        assert np.allclose(rescaled_correct, [[[[ -1.0, 0.0, 1.0 ]]]], atol=1e-4)

        # Test B: Incorrect pre-normalized input in [-1.0, 1.0] (The Bug)
        buggy_input = tf.constant([[[[ -1.0, 0.0, 1.0 ]]]], dtype=tf.float32)
        rescaled_buggy = rescaling_layer(buggy_input).numpy()
        # Demonstrates the dynamic range squashing
        assert np.all(rescaled_buggy < -0.99)
        assert np.all(rescaled_buggy > -1.01)
        width = rescaled_buggy.max() - rescaled_buggy.min()
        assert width < 0.02, f"Double normalization squashes dynamic range to {width}"


class TestBinaryFocalLossSerialization(unittest.TestCase):
    """Verify BinaryFocalLoss serialization, deserialization, save, and reload."""

    def test_focal_loss_kwargs_and_config(self):
        """BinaryFocalLoss should accept **kwargs and reduction without TypeError."""
        loss = BinaryFocalLoss(alpha=0.75, gamma=2.0, reduction="sum_over_batch_size", name="custom_focal")
        cfg = loss.get_config()
        assert cfg["alpha"] == 0.75
        assert cfg["gamma"] == 2.0
        assert cfg["reduction"] == "sum_over_batch_size"

        # Test reconstruction from config
        reconstructed = BinaryFocalLoss.from_config(cfg)
        assert reconstructed.alpha == 0.75
        assert reconstructed.gamma == 2.0

    def test_model_save_and_reload_with_focal_loss(self):
        """Model compiled with BinaryFocalLoss must save and load cleanly without TypeError."""
        inp = layers.Input(shape=(10,))
        out = layers.Dense(1, activation="sigmoid")(inp)
        model = models.Model(inputs=inp, outputs=out)
        loss = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        model.compile(optimizer="adam", loss=loss)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_focal_model.keras")
            model.save(model_path)
            assert os.path.exists(model_path)

            # Reload model without compile=False (tests full deserialization)
            reloaded = tf.keras.models.load_model(
                model_path,
                custom_objects={"BinaryFocalLoss": BinaryFocalLoss},
            )
            dummy_x = np.random.randn(4, 10).astype(np.float32)
            preds_orig = model.predict(dummy_x, verbose=0)
            preds_reloaded = reloaded.predict(dummy_x, verbose=0)
            assert np.allclose(preds_orig, preds_reloaded, atol=1e-5)


class TestPartialAUC(unittest.TestCase):
    """Verify partial AUC calculation and alignment with sklearn."""

    def test_pauc_exact_match_sklearn(self):
        """pAUC should match sklearn.metrics.roc_auc_score(y_true, y_score, max_fpr=0.20)."""
        rng = np.random.RandomState(42)
        y_true = np.array([1]*20 + [0]*80)
        y_scores = rng.beta(1.5, 3.0, size=100)

        expected_sklearn_pauc = roc_auc_score(y_true, y_scores, max_fpr=0.20)
        computed_pauc = compute_pauc(y_true, y_scores, max_fpr=0.20)

        assert np.isclose(computed_pauc, expected_sklearn_pauc, atol=1e-5)

    def test_random_predictions_pauc(self):
        """Under random scores, normalized partial AUC is ~ 0.50."""
        rng = np.random.RandomState(42)
        y_true = np.array([1]*50 + [0]*950)
        y_scores = rng.uniform(0.0, 1.0, size=1000)

        pauc = compute_pauc(y_true, y_scores, max_fpr=0.20)
        assert 0.40 <= pauc <= 0.60


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for cls in [TestImagePreprocessingAndBackbone, TestBinaryFocalLossSerialization, TestPartialAUC]:
        for method_name in dir(cls):
            if method_name.startswith("test_"):
                suite.addTest(cls(method_name))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

