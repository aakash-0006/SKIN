"""Stage J: Dedicated test suite verifying Candidate 1B normal-hand appearance detector integration."""
import os
import unittest
import hashlib
from pathlib import Path
import numpy as np

from src.normal_hand.cropper import HandCropper
from src.normal_hand.feature_extractor import HandFeatureExtractor
from src.normal_hand.anomaly_detector import MahalanobisAnomalyDetector
from src.normal_hand.pipeline import NormalHandPipeline
from src.predict import (
    get_normal_hand_pipeline,
    predict_bytes,
    PROJECT_DIR,
)
import src.predict as predict_module


class TestStageJCandidate1BIntegration(unittest.TestCase):
    """Test suite ensuring Candidate 1B is properly wired, reproduces fitting configuration, and maintains safety."""

    @classmethod
    def setUpClass(cls):
        cls.candidate_path = (
            PROJECT_DIR
            / "models"
            / "experiments"
            / "normal_hand_v1"
            / "stage_g"
            / "candidate_1b_l2_mahalanobis_k32.npz"
        )
        cls.baseline_path = (
            PROJECT_DIR
            / "models"
            / "experiments"
            / "normal_hand_v1"
            / "hand_anomaly_baseline.npz"
        )

    def test_1_candidate_1b_artifact_integrity_and_hyperparameters(self):
        """Rule: Candidate 1B artifact must exist, have k=32, shrinkage=0.05, and threshold=6.931."""
        self.assertTrue(self.candidate_path.exists(), f"Candidate 1B artifact missing at {self.candidate_path}")
        
        # Verify SHA-256
        h = hashlib.sha256()
        with open(self.candidate_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self.assertEqual(
            h.hexdigest().upper(),
            "C4F0FED96CB0308C8155CED51607E0DF872F4C91EC2180ADCD4ED311267FC5AB",
            "Candidate 1B hash mismatch!"
        )

        detector = MahalanobisAnomalyDetector.load(str(self.candidate_path))
        self.assertEqual(detector.n_components, 32, "Candidate 1B must use k=32 PCA dimensions")
        self.assertAlmostEqual(detector.regularization, 0.05, places=3, msg="Candidate 1B shrinkage must be 0.05")
        self.assertAlmostEqual(detector.threshold, 6.931, places=3, msg="Candidate 1B threshold must be 6.931")
        self.assertEqual(detector.pca.components_.shape, (32, 576), "PCA projection matrix must be (32, 576)")
        self.assertEqual(detector.inv_covariance.shape, (32, 32), "Precision matrix must be (32, 32)")

    def test_2_get_normal_hand_pipeline_loads_candidate_1b_with_l2_normalization(self):
        """Rule: get_normal_hand_pipeline must load Candidate 1B with normalize_features=True."""
        pipeline = get_normal_hand_pipeline()
        self.assertIsInstance(pipeline, NormalHandPipeline)
        self.assertTrue(pipeline.normalize_features, "Candidate 1B requires normalize_features=True")
        self.assertEqual(pipeline.anomaly_detector.n_components, 32)
        self.assertAlmostEqual(pipeline.anomaly_detector.threshold, 6.931, places=3)

    def test_3_feature_extractor_l2_normalization(self):
        """Rule: HandFeatureExtractor must produce unit-norm vectors when normalize=True."""
        fe = HandFeatureExtractor(target_size=(224, 224))
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        feats_norm = fe.extract_features(dummy, normalize=True)
        self.assertEqual(feats_norm.shape, (1, 576))
        norm_val = np.linalg.norm(feats_norm, axis=-1)[0]
        self.assertAlmostEqual(norm_val, 1.0, places=5, msg="L2-normalized feature must have unit norm")

        feats_raw = fe.extract_features(dummy, normalize=False)
        raw_norm = np.linalg.norm(feats_raw, axis=-1)[0]
        self.assertNotAlmostEqual(raw_norm, 1.0, places=2, msg="Raw features should not have unit norm")

    def test_4_candidate_1b_scoring_consistency(self):
        """Rule: Candidate 1B scoring must produce positive Mahalanobis distances on random normalized features."""
        detector = MahalanobisAnomalyDetector.load(str(self.candidate_path))
        rng = np.random.default_rng(123)
        X = rng.normal(size=(5, 576)).astype(np.float32)
        X /= np.linalg.norm(X, axis=-1, keepdims=True)
        scores = detector.score(X)
        self.assertEqual(len(scores), 5)
        self.assertTrue((scores >= 0.0).all(), "Mahalanobis distances must be non-negative")

    def test_5_feature_flag_is_disabled_by_default(self):
        """Rule: ENABLE_NORMAL_HAND_DETECTOR must be False by default."""
        self.assertFalse(
            predict_module.ENABLE_NORMAL_HAND_DETECTOR,
            "ENABLE_NORMAL_HAND_DETECTOR must remain False until explicit approval."
        )

    def test_6_disabled_mode_hand_suppression(self):
        """Rule: In disabled mode, non-dermoscopic hand photos must receive NO condition label or disease scores."""
        fixture_hand = PROJECT_DIR / "tests" / "fixtures" / "webcam_classroom_hand.jpg"
        if not fixture_hand.exists():
            self.skipTest("Fixture webcam_classroom_hand.jpg not found.")

        data = fixture_hand.read_bytes()
        res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        
        # In disabled mode, classroom hand is either rejected by domain gate or suppressed by hand gate
        self.assertEqual(res.get("code"), None, "Hand must NOT receive disease code")
        self.assertEqual(res.get("scores", []), [], "Hand must NOT receive disease scores")
        self.assertFalse(res.get("melanoma_flag", False), "Hand must NOT receive melanoma flag")
        self.assertNotIn("score", res, "Must NOT produce a disease score")
        self.assertIsNone(res.get("hand_analysis"), "hand_analysis must be None when flag is False")

    def test_7_enabled_mode_hand_inspection_with_candidate_1b(self):
        """Rule: In enabled mode, hand photos use Candidate 1B threshold (6.931) and produce cautious taxonomy."""
        fixture_hand = PROJECT_DIR / "tests" / "fixtures" / "webcam_classroom_hand.jpg"
        if not fixture_hand.exists():
            self.skipTest("Fixture webcam_classroom_hand.jpg not found.")

        data = fixture_hand.read_bytes()
        predict_module.ENABLE_NORMAL_HAND_DETECTOR = True
        try:
            res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        finally:
            predict_module.ENABLE_NORMAL_HAND_DETECTOR = False

        self.assertTrue(res.get("is_hand", False))
        self.assertEqual(res.get("code"), None)
        self.assertEqual(res.get("scores", []), [])
        self.assertFalse(res.get("melanoma_flag", False))
        
        hand_an = res.get("hand_analysis")
        self.assertIsNotNone(hand_an, "hand_analysis must be present when flag is True")
        self.assertAlmostEqual(hand_an["threshold"], 6.931, places=3, msg="Threshold must be Candidate 1B (6.931)")
        self.assertIn(
            hand_an["decision"],
            {"USABLE_HAND_IMAGE", "UNSUITABLE_OR_UNCERTAIN_INPUT", "UNUSUAL_RELATIVE_TO_TRAINING_DATA"}
        )
        self.assertNotIn("DISEASE", hand_an["message"].upper())
        self.assertNotIn("HEALTHY", hand_an["message"].upper())

    def test_8_non_hand_fixture_rejected(self):
        """Rule: Non-hand gaming poster must be rejected at input validation stage."""
        poster = PROJECT_DIR / "tests" / "fixtures" / "gaming_poster.jpg"
        if not poster.exists():
            self.skipTest("Fixture gaming_poster.jpg not found.")

        data = poster.read_bytes()
        res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        self.assertFalse(res.get("ok", True), "Gaming poster must be rejected")
        self.assertEqual(res.get("code"), None)
        self.assertEqual(res.get("scores", []), [])

    def test_9_protected_asset_hashes_unchanged(self):
        """Rule: Protected models, OOD gate, splits, config, and baseline must match locked hashes."""
        assets = {
            "models/best_model.keras": "C6AF0869563F708D1DC26E67668F0838D7EDFD277B7B585FA65527BCFFDDF2FD",
            "models/isic_finetuned.keras": "6149DBD49D5DC403CF81E55BA118B4CD20D70D4F7A0098617BAE994962005FF2",
            "models/isic_augmented_v1.keras": "FE3EB872F919BCDAA4AE69C318993B50A33C0FEE815D00AADD2BAB75ADA99D34",
            "models/ood_gate.npz": "DCD9974A82E40E4DECFB1170893B7934C9BAB9B05C9DF316046F25EA47479DBE",
            "outputs/splits.csv": "8DEEDC0338273B81EAD6BC89D8C12D53E6091906390EC81C5EC9CE37170CB3EC",
            "outputs/combined_splits.csv": "07F94433D6BB638EDE552EF9338F0E2258BEAFBEFB85F487CDC58A6DD7548149",
            "config.json": "AD23CC09A5050C5B390AF56BCE25B389CAC72E1BB52E3416F8A48C08801DEDB1",
            "models/experiments/normal_hand_v1/hand_anomaly_baseline.npz": "5B2B53451932ACC9644AB612AA0AF7FF0033E8A03ED4503C8D2EF686DCB8DBF4",
        }
        for rel_path, expected_hash in assets.items():
            full_path = PROJECT_DIR / rel_path
            self.assertTrue(full_path.exists(), f"Protected file missing: {rel_path}")
            h = hashlib.sha256()
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            self.assertEqual(h.hexdigest().upper(), expected_hash, f"Hash mismatch for {rel_path}!")


if __name__ == "__main__":
    unittest.main()
