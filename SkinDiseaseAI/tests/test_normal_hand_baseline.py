"""Unit tests for src/normal_hand modules (Cropper, Feature Extractor, Anomaly Detector, Pipeline)."""
import os
import unittest
import numpy as np

from src.normal_hand.cropper import HandCropper, CropResult
from src.normal_hand.feature_extractor import HandFeatureExtractor
from src.normal_hand.anomaly_detector import MahalanobisAnomalyDetector, KNNAnomalyDetector
from src.normal_hand.pipeline import NormalHandPipeline, HandInspectionResult


class TestNormalHandBaseline(unittest.TestCase):
    """Test suite ensuring normal hand baseline utilities function safely and produce cautious outputs."""

    def test_cropper_rejects_empty_or_degenerate_input(self):
        cropper = HandCropper()
        res_empty = cropper.crop_hand(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertFalse(res_empty.success)
        self.assertEqual(res_empty.failure_reason, "EMPTY_OR_INVALID_IMAGE")

        res_tiny = cropper.crop_hand(np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertFalse(res_tiny.success)
        self.assertEqual(res_tiny.failure_reason, "RESOLUTION_TOO_LOW")

    def test_cropper_rejects_non_hand_image(self):
        cropper = HandCropper()
        # Synthetic noise image without hand
        noise = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        res = cropper.crop_hand(noise)
        self.assertFalse(res.success)
        self.assertEqual(res.failure_reason, "NO_HAND_DETECTED")

    def test_feature_extractor_shape(self):
        fe = HandFeatureExtractor()
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = fe.extract_features(dummy)
        self.assertEqual(feats.shape, (1, 576))
        self.assertEqual(feats.dtype, np.float32)

    def test_mahalanobis_fit_and_serialization(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(100, 576)).astype(np.float32)
        X_val = rng.normal(size=(30, 576)).astype(np.float32)

        det = MahalanobisAnomalyDetector(n_components=16, regularization=0.05)
        det.fit(X_train)
        thr = det.calibrate_threshold(X_val, alpha=0.05)
        self.assertGreater(thr, 0.0)

        # Test distance scoring
        scores = det.score(X_val)
        self.assertEqual(len(scores), 30)
        self.assertTrue((scores >= 0.0).all())

        # Test save and reload
        tmp_path = "scratch/temp_detector.npz"
        os.makedirs("scratch", exist_ok=True)
        det.save(tmp_path)

        loaded = MahalanobisAnomalyDetector.load(tmp_path)
        self.assertEqual(loaded.threshold, det.threshold)
        scores_loaded = loaded.score(X_val)
        np.testing.assert_allclose(scores, scores_loaded, rtol=1e-5)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def test_pipeline_output_vocabulary_strictly_cautious(self):
        """Verifies pipeline output decision vocabulary contains no clinical disease claims."""
        cropper = HandCropper()
        fe = HandFeatureExtractor()
        det = MahalanobisAnomalyDetector(n_components=8, regularization=0.05)
        X_dummy = np.random.randn(50, 576).astype(np.float32)
        det.fit(X_dummy)
        det.threshold = 5.0

        pipe = NormalHandPipeline(cropper=cropper, feature_extractor=fe, anomaly_detector=det)

        # Non-hand image
        res_nonhand = pipe.inspect_image(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertIn(
            res_nonhand.decision,
            {"USABLE_HAND_IMAGE", "UNSUITABLE_OR_UNCERTAIN_INPUT", "UNUSUAL_RELATIVE_TO_TRAINING_DATA"}
        )
        self.assertNotIn("DISEASE", res_nonhand.decision)
        self.assertNotIn("HEALTHY", res_nonhand.decision)
        self.assertNotIn("NO_DISEASE", res_nonhand.decision)


if __name__ == "__main__":
    unittest.main()
