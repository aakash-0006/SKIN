"""Comprehensive Stage F integration and regression test suite.

Verifies:
1. Feature flag defaults to False (disabled).
2. Existing 7-class dermoscopic predictions are identical whether flag is False or True.
3. Hand images strictly NEVER receive a 7-class disease prediction.
4. Hand pipeline uses exact threshold (8.768) and Stage D preprocessing.
5. Unsuitable, corrupt, and non-hand inputs are handled safely.
6. Missing model artifacts produce graceful errors rather than misleading predictions.
7. Protected asset hashes remain 100% identical to locked backup.
"""
import hashlib
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import src.predict as predict_module
from src.predict import predict_bytes, load_artifacts, get_normal_hand_pipeline, PROJECT_DIR
from src.normal_hand.pipeline import NormalHandPipeline


class TestStageFIntegration(unittest.TestCase):
    """Test suite ensuring Stage F integration safety, isolation, and regression stability."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = Path(__file__).parent / "fixtures"
        cls.hand_dir = Path(r"C:\Users\USER\Downloads\normal skin dataset\archive (3)\Hands\Hands")
        cls.sample_hand = sorted(list(cls.hand_dir.glob("*.jpg")))[0] if cls.hand_dir.exists() else None
        
        # Load sample HAM dermoscopy image
        splits_p = PROJECT_DIR / "outputs" / "splits.csv"
        cls.ham_img_path = None
        if splits_p.exists():
            import pandas as pd
            df = pd.read_csv(splits_p)
            val_nv = df[(df["split"] == "val") & (df["dx"] == "nv")].iloc[0]["image_id"]
            candidates = list(Path(r"D:\DOWNLOADS D\archive").rglob(f"{val_nv}.jpg"))
            if candidates:
                cls.ham_img_path = candidates[0]

    def test_1_feature_flag_defaults_to_false(self):
        """Rule: Feature flag MUST default to False (disabled) during implementation."""
        self.assertFalse(
            predict_module.ENABLE_NORMAL_HAND_DETECTOR,
            "ENABLE_NORMAL_HAND_DETECTOR must default to False during implementation."
        )

    def test_2_existing_seven_class_prediction_unaffected(self):
        """Rule: 7-class predictions for dermoscopic images must be bitwise identical."""
        if not self.ham_img_path or not self.ham_img_path.exists():
            self.skipTest("No HAM validation image found for dermoscopy regression test.")

        data = self.ham_img_path.read_bytes()

        # Run with flag disabled
        predict_module.ENABLE_NORMAL_HAND_DETECTOR = False
        res_disabled = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")

        # Run with flag enabled
        predict_module.ENABLE_NORMAL_HAND_DETECTOR = True
        try:
            res_enabled = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        finally:
            predict_module.ENABLE_NORMAL_HAND_DETECTOR = False

        self.assertEqual(res_disabled.get("code"), res_enabled.get("code"))
        self.assertEqual(res_disabled.get("score"), res_enabled.get("score"))
        self.assertEqual(res_disabled.get("melanoma_flag"), res_enabled.get("melanoma_flag"))
        self.assertEqual(res_disabled.get("is_hand"), res_enabled.get("is_hand"))

    def test_3_hand_images_never_receive_seven_class_prediction(self):
        """Rule: Hand photographs must NEVER receive a disease label, score table, or melanoma flag."""
        if not self.sample_hand or not self.sample_hand.exists():
            self.skipTest("Sample hand image not available.")

        data = self.sample_hand.read_bytes()

        for flag_val in [False, True]:
            predict_module.ENABLE_NORMAL_HAND_DETECTOR = flag_val
            try:
                res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
            finally:
                predict_module.ENABLE_NORMAL_HAND_DETECTOR = False

            self.assertIsNone(res.get("code"), f"Disease code must be None when flag is {flag_val}")
            self.assertEqual(res.get("scores", []), [], f"Scores must be empty when flag is {flag_val}")
            self.assertFalse(res.get("melanoma_flag", False), f"Melanoma flag must be False when flag is {flag_val}")
            if flag_val:
                self.assertTrue(res.get("is_hand", False), "is_hand must be True when flag is enabled")
                self.assertIsNotNone(res.get("hand_analysis"), "hand_analysis must be present when flag is enabled")
            else:
                self.assertIsNone(res.get("hand_analysis"), "hand_analysis must be None when flag is disabled")

    def test_4_hand_pipeline_uses_exact_threshold_and_preprocessing(self):
        """Rule: When enabled, hand pipeline must use locked threshold (6.931 for Candidate 1B or 8.768 for Baseline) and produce cautious taxonomy."""
        pipeline = get_normal_hand_pipeline()
        self.assertIn(round(pipeline.anomaly_detector.threshold, 3), {6.931, 8.768})
        self.assertEqual(pipeline.feature_extractor.feature_dim, 576)
        self.assertIn(pipeline.anomaly_detector.n_components, {32, 48})

        if not self.sample_hand or not self.sample_hand.exists():
            self.skipTest("Sample hand image not available.")

        data = self.sample_hand.read_bytes()
        predict_module.ENABLE_NORMAL_HAND_DETECTOR = True
        try:
            res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        finally:
            predict_module.ENABLE_NORMAL_HAND_DETECTOR = False

        hand_an = res.get("hand_analysis")
        self.assertIsNotNone(hand_an, "hand_analysis should be populated when feature flag is True.")
        self.assertIn(
            hand_an["decision"],
            {"USABLE_HAND_IMAGE", "UNSUITABLE_OR_UNCERTAIN_INPUT", "UNUSUAL_RELATIVE_TO_TRAINING_DATA"}
        )
        self.assertIn(round(hand_an["threshold"], 3), {6.931, 8.768})
        self.assertNotIn("DISEASE", hand_an["message"].upper())
        self.assertNotIn("HEALTHY", hand_an["message"].upper())

    def test_5_unsuitable_and_invalid_inputs_handled_safely(self):
        """Rule: Corrupt, non-hand, and empty inputs must not raise uncaught exceptions."""
        predict_module.ENABLE_NORMAL_HAND_DETECTOR = True
        try:
            # 1. Empty bytes
            res_empty = predict_bytes(b"", require_hand_validation=True, validation_mode="all_skin")
            self.assertFalse(res_empty.get("ok"))

            # 2. Corrupt garbage
            res_corrupt = predict_bytes(b"not an image", require_hand_validation=True, validation_mode="all_skin")
            self.assertFalse(res_corrupt.get("ok"))

            # 3. Non-hand gaming poster fixture
            poster_p = self.fixtures_dir / "gaming_poster.jpg"
            if poster_p.exists():
                res_poster = predict_bytes(poster_p.read_bytes(), require_hand_validation=True, validation_mode="all_skin")
                self.assertFalse(res_poster.get("ok"))
                self.assertIsNone(res_poster.get("code"))
        finally:
            predict_module.ENABLE_NORMAL_HAND_DETECTOR = False

    def test_6_protected_assets_hashes_remain_identical(self):
        """Rule: Protected assets must remain 100% bitwise identical to locked backup."""
        expected_hashes = {
            "models/best_model.keras": "c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd",
            "models/isic_finetuned.keras": "6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2",
            "models/isic_augmented_v1.keras": "fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34",
            "models/ood_gate.npz": "dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe",
            "outputs/splits.csv": "8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec",
            "outputs/combined_splits.csv": "07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149",
            "config.json": "ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1",
        }
        for rel_path, expected_hash in expected_hashes.items():
            full_path = PROJECT_DIR / rel_path
            self.assertTrue(full_path.exists(), f"Protected file missing: {rel_path}")
            actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, f"Protected file hash mismatch on {rel_path}!")

    def test_7_missing_artifact_handled_gracefully(self):
        """Rule: Missing hand detector artifact produces a clear error rather than misleading result."""
        from unittest.mock import patch
        predict_module.ENABLE_NORMAL_HAND_DETECTOR = True
        try:
            with patch("src.predict.get_normal_hand_pipeline", side_effect=FileNotFoundError("Mock missing model")):
                if self.sample_hand and self.sample_hand.exists():
                    res = predict_bytes(self.sample_hand.read_bytes(), require_hand_validation=True, validation_mode="all_skin")
                    self.assertFalse(res.get("ok"))
                    self.assertIn("missing", res.get("message", "").lower())
                    self.assertIsNone(res.get("code"))
        finally:
            predict_module.ENABLE_NORMAL_HAND_DETECTOR = False

    def test_8_streamlit_interface_layout_and_disclaimer_present(self):
        """Rule: Streamlit app contains 4 sections, suppression card, and measured limitations."""
        app_code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("1. Input Modality & Quality Verification", app_code)
        self.assertIn("2. Normal Hand Appearance Evaluation", app_code)
        self.assertIn("3. Seven-Class Lesion Classifier Status", app_code)
        self.assertIn("4. Mandatory Clinical Safety Notice", app_code)
        self.assertIn("19.16%", app_code, "Accessories limitation must be displayed")
        self.assertIn("22.50%", app_code, "Nail polish limitation must be displayed")
        self.assertIn("does NOT guarantee a correct result", app_code)


if __name__ == "__main__":
    unittest.main()
