"""Regression tests for inputs that previously failed or slipped past validation."""
import glob
import os
import unittest

from src.gate import check_image
from src.predict import predict_bytes


class TestRegressionFixtures(unittest.TestCase):
    """Verifies that non-dermoscopic camera photos (webcam arm, gaming poster) are rejected."""

    def setUp(self):
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    def test_fixtures_exist_and_rejected(self):
        if not os.path.exists(self.fixtures_dir):
            self.skipTest(f"Fixtures directory {self.fixtures_dir} does not exist yet. Please place the webcam arm and poster files in tests/fixtures/")

        # Locate webcam arm and poster fixture files
        arm_files = [f for f in os.listdir(self.fixtures_dir) if "arm" in f.lower() or "webcam" in f.lower()]
        poster_files = [f for f in os.listdir(self.fixtures_dir) if "poster" in f.lower()]

        if not arm_files and not poster_files:
            self.skipTest("No webcam arm or poster image files found in tests/fixtures/. Please place the two fixture files in tests/fixtures/")

        for fname in arm_files + poster_files:
            fpath = os.path.join(self.fixtures_dir, fname)
            with self.subTest(fixture=fname):
                with open(fpath, "rb") as f:
                    data = f.read()

                # Test 1: Direct Domain Gate Check must reject
                gate_res = check_image(data)
                self.assertEqual(gate_res["status"], "rejected", f"Domain gate should have rejected fixture {fname}")

                # Test 2: Full predict_bytes pipeline must reject with ok=False
                pred_res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
                self.assertFalse(pred_res["ok"], f"predict_bytes should have rejected fixture {fname}")
                self.assertNotIn("score", pred_res, f"No classification score should be produced for fixture {fname}")
                self.assertFalse(pred_res.get("melanoma_flag", False), f"Melanoma flag must not trigger for fixture {fname}")

    def test_webcam_classroom_photo_produces_no_label(self):
        """Verifies that a webcam classroom photo produces NO condition label, NO score table, and NO melanoma flag."""
        classroom_path = os.path.join(self.fixtures_dir, "webcam_classroom_hand.jpg")
        if not os.path.exists(classroom_path):
            self.skipTest("webcam_classroom_hand.jpg fixture missing")
        with open(classroom_path, "rb") as f:
            data = f.read()

        # Run pipeline in standard dermoscopy mode
        res = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        self.assertNotIn("score", res, "Must NOT produce a disease score")
        self.assertIsNone(res.get("code"), "Must NOT produce a disease class label")
        self.assertEqual(res.get("scores", []), [], "Score table must be empty")
        self.assertFalse(res.get("melanoma_flag", False), "Melanoma flag must not trigger")
        self.assertTrue(
            "outside the validated domain" in res.get("message", "") or
            "Hand detected" in res.get("message", ""),
            f"Expected domain or hand warning message, got: {res.get('message')}"
        )


if __name__ == "__main__":
    unittest.main()
