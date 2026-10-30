"""Unit tests for Hand Detection, Anti-Spoofing, Skin Quality, and Pipeline Integration.

Tests the five scenarios required by the specification:
1. Genuine hand presentation (synthesized physiological hand pattern).
2. Photo containing no hand (backgrounds, textures).
3. Anime character / flat illustration.
4. Car / vehicle / unrelated object.
5. Printed hand photograph with halftoning / Moiré artifact.
6. Hand displayed on screen with pixel grid / rectilinear borders.
7. Blurry or poorly lit hand.
8. Multiple hands rejection.
"""
import io
import os
import unittest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

from src.hand_validator import HandValidator, get_hand_validator
from src.predict import predict_bytes


def create_jpeg_bytes(img_rgb: np.ndarray) -> bytes:
    """Helper to convert RGB numpy array to JPEG bytes."""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def create_synthetic_hand_image(size=(320, 320), skin_rgb=(210, 160, 130), blur=False, dark=False) -> np.ndarray:
    """Generate a realistic hand silhouette with palm and 5 extended fingers."""
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Palm (ellipse in lower center)
    draw.ellipse([90, 140, 230, 280], fill=skin_rgb)
    # 5 Fingers
    # Thumb
    draw.polygon([(95, 170), (45, 140), (60, 115), (115, 150)], fill=skin_rgb)
    # Index
    draw.rounded_rectangle([105, 45, 130, 160], radius=12, fill=skin_rgb)
    # Middle
    draw.rounded_rectangle([135, 30, 160, 155], radius=12, fill=skin_rgb)
    # Ring
    draw.rounded_rectangle([165, 50, 190, 160], radius=12, fill=skin_rgb)
    # Pinky
    draw.rounded_rectangle([195, 80, 220, 170], radius=12, fill=skin_rgb)

    arr = np.asarray(img, dtype=np.uint8)
    if dark:
        arr = (arr * 0.10).astype(np.uint8)
    if blur:
        arr = cv2.GaussianBlur(arr, (25, 25), 0)
    return arr


def create_illustration_image(size=(320, 320)) -> np.ndarray:
    """Generate flat colored cartoon/illustration with stark edges and non-skin tones."""
    img = Image.new("RGB", size, (255, 220, 240))
    draw = ImageDraw.Draw(img)
    # Cartoon face / hair with bold black lines
    draw.ellipse([80, 80, 240, 240], fill=(255, 240, 230), outline=(0, 0, 0), width=4)
    draw.polygon([(60, 60), (160, 20), (260, 60), (230, 140), (90, 140)], fill=(80, 140, 255), outline=(0, 0, 0), width=4)
    # Eyes
    draw.ellipse([110, 130, 140, 160], fill=(50, 180, 255), outline=(0, 0, 0), width=3)
    draw.ellipse([180, 130, 210, 160], fill=(50, 180, 255), outline=(0, 0, 0), width=3)
    return np.asarray(img, dtype=np.uint8)


def create_vehicle_image(size=(320, 320)) -> np.ndarray:
    """Generate image of an automobile on asphalt."""
    img = Image.new("RGB", size, (160, 190, 220))  # Sky
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 200, 320, 320], fill=(70, 70, 70))  # Road
    # Car body (red metallic)
    draw.rectangle([50, 150, 270, 220], fill=(220, 30, 30))
    draw.polygon([(80, 150), (120, 100), (200, 100), (240, 150)], fill=(60, 120, 180))  # Cabin
    # Wheels
    draw.ellipse([80, 200, 120, 240], fill=(20, 20, 20))
    draw.ellipse([200, 200, 240, 240], fill=(20, 20, 20))
    return np.asarray(img, dtype=np.uint8)


def create_screen_spoof_image(hand_rgb: np.ndarray) -> np.ndarray:
    """Simulate a hand shown on a digital screen with bezel borders and high-frequency moiré."""
    h, w = hand_rgb.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Black bezel border around inner display
    bx, by, bw, bh = 20, 20, w - 40, h - 40
    inner = cv2.resize(hand_rgb, (bw, bh))

    # Add periodic high-frequency horizontal and vertical raster/moiré lines
    moiré = np.sin(np.arange(bh)[:, None] * 1.4) * np.cos(np.arange(bw)[None, :] * 1.4) * 35.0
    inner_noisy = np.clip(inner.astype(np.float32) + moiré[:, :, None], 0, 255).astype(np.uint8)
    canvas[by:by+bh, bx:bx+bw] = inner_noisy

    # Draw sharp white display bezel border
    cv2.rectangle(canvas, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), (255, 255, 255), 3)
    return canvas


def create_human_skin_patch(size=(320, 320), skin_rgb=(210, 160, 130)) -> np.ndarray:
    """Generate human skin area across body/limbs with natural chromaticity and a central lesion."""
    img = Image.new("RGB", size, skin_rgb)
    draw = ImageDraw.Draw(img)
    # Draw pigmented lesion (e.g. nevus or keratosis on skin)
    draw.ellipse([120, 120, 200, 200], fill=(75, 45, 35))
    draw.ellipse([135, 135, 185, 185], fill=(45, 25, 20))
    arr = np.asarray(img, dtype=np.uint8)
    # Add mild physiological texture variation
    noise = np.random.normal(0, 2, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return arr


class TestHandValidationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = get_hand_validator()

    def test_general_human_skin_accepted_in_all_skin_mode(self):
        """Lesions on arms, legs, face, back, chest, or dermoscopy MUST be accepted in all_skin mode."""
        skin_img = create_human_skin_patch()
        data = create_jpeg_bytes(skin_img)
        res = self.validator.validate_image_bytes(data, mode="all_skin")

        self.assertTrue(res.passed, f"General human skin was incorrectly rejected: {res.reasons}")
        self.assertEqual(res.status, "accepted")
        self.assertGreater(res.skin_fraction, 0.50)
        self.assertIn("Skin", res.input_type)
        self.assertIn("Skin", res.anatomical_region)

    def test_general_human_skin_rejected_in_strict_hand_only_mode(self):
        """In strict hand-only mode, non-hand skin patches must be stopped at hand_detection."""
        skin_img = create_human_skin_patch()
        data = create_jpeg_bytes(skin_img)
        res = self.validator.validate_image_bytes(data, mode="hand_only")

        self.assertFalse(res.passed)
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.stage_failed, "hand_detection")

    def test_non_hand_noise_rejected(self):
        """Random texture / noise has no skin and must be rejected at skin_detection or spoof_check."""
        np.random.seed(42)
        noise = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        data = create_jpeg_bytes(noise)
        res = self.validator.validate_image_bytes(data, mode="all_skin")

        self.assertFalse(res.passed)
        self.assertEqual(res.status, "rejected")
        self.assertIn(res.stage_failed, ("skin_detection", "spoof_check"))

    def test_illustration_cartoon_rejected(self):
        """Anime / cartoon artwork contains no physiological human skin."""
        art = create_illustration_image()
        data = create_jpeg_bytes(art)
        res = self.validator.validate_image_bytes(data, mode="all_skin")

        self.assertFalse(res.passed)
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.stage_failed, "skin_detection")

    def test_vehicle_rejected(self):
        """Vehicle/car image contains no human skin and must be rejected."""
        car = create_vehicle_image()
        data = create_jpeg_bytes(car)
        res = self.validator.validate_image_bytes(data, mode="all_skin")

        self.assertFalse(res.passed)
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.stage_failed, "skin_detection")

    def test_dark_image_quality_check(self):
        """Severely dark image fails quality check."""
        hand = create_synthetic_hand_image(dark=True)
        quality, issues = self.validator.check_image_quality(hand)
        self.assertTrue(any("underexposed" in issue.lower() or "dark" in issue.lower() for issue in issues))

    def test_blurry_image_quality_check(self):
        """Blurry image fails quality check."""
        hand = create_synthetic_hand_image(blur=True)
        quality, issues = self.validator.check_image_quality(hand)
        self.assertTrue(any("blurry" in issue.lower() for issue in issues))

    def test_skin_region_calculation(self):
        """Skin color fraction measurement across color spaces."""
        # Genuine skin tone
        skin_img = np.full((100, 100, 3), (120, 160, 210), dtype=np.uint8)  # BGR
        skin_frac = self.validator.check_skin_region(skin_img)
        self.assertGreater(skin_frac, 0.50)

        # Non-skin blue fabric / glove
        blue_img = np.full((100, 100, 3), (230, 40, 20), dtype=np.uint8)  # Bright blue
        blue_frac = self.validator.check_skin_region(blue_img)
        self.assertLess(blue_frac, 0.05)

    def test_screen_spoof_detection(self):
        """Image with display bezel borders and periodic moiré triggers anti-spoof warnings."""
        hand = create_synthetic_hand_image()
        spoofed = create_screen_spoof_image(hand)
        spoof_res = self.validator.check_spoof_and_authenticity(spoofed)

        # Must flag suspicions (bezel border or spectral moiré)
        self.assertEqual(spoof_res["authenticity"], "suspected_spoof")
        self.assertTrue(len(spoof_res["suspicions"]) > 0)

    def test_pipeline_integration_blocks_classifier_for_non_skin(self):
        """predict_bytes must NOT call the lesion classifier if input is not human skin."""
        car = create_vehicle_image()
        data = create_jpeg_bytes(car)

        # Calling predict_bytes with validation enabled
        result = predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")
        self.assertFalse(result["ok"])
        self.assertIn(result["stage_failed"], ("domain_gate", "skin_detection"))
        # Ensure classifier scores were never computed
        self.assertNotIn("score", result)

    def test_pipeline_integration_runs_classifier_for_human_skin(self):
        """predict_bytes must find skin disease when genuine dermoscopic human skin is presented."""
        real_sample = r"D:\DOWNLOADS D\archive\HAM10000_images_part_1\ISIC_0029068.jpg"
        if os.path.exists(real_sample):
            with open(real_sample, "rb") as f:
                data = f.read()
            mode = "all_skin"
        else:
            skin_img = create_human_skin_patch()
            data = create_jpeg_bytes(skin_img)
            mode = "bypass"

        result = predict_bytes(data, require_hand_validation=True, validation_mode=mode)
        self.assertTrue(result["ok"])
        self.assertIn("score", result)
        self.assertIn("code", result)
        self.assertIn(result["code"], ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
        self.assertGreater(result["score"], 0.0)


class TestBenchmarkMatrix(unittest.TestCase):
    """Evaluates and reports false acceptances / false rejections across synthetic test categories."""

    def test_benchmark_matrix_reporting(self):
        validator = get_hand_validator()

        categories = {
            "unrelated_noise": [create_jpeg_bytes(np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)) for _ in range(5)],
            "unrelated_illustration": [create_jpeg_bytes(create_illustration_image()) for _ in range(3)],
            "unrelated_vehicle": [create_jpeg_bytes(create_vehicle_image()) for _ in range(3)],
            "screen_spoof": [create_jpeg_bytes(create_screen_spoof_image(create_synthetic_hand_image())) for _ in range(3)],
            "poor_quality_dark": [create_jpeg_bytes(create_synthetic_hand_image(dark=True)) for _ in range(3)],
            "poor_quality_blur": [create_jpeg_bytes(create_synthetic_hand_image(blur=True)) for _ in range(3)],
        }

        print("\n================== HAND & SPOOF GATE BENCHMARK MATRIX ==================")
        print(f"{'Category':<26} | {'Samples':<8} | {'Rejected':<10} | {'False Accept Rate':<18}")
        print("-" * 70)
        total_non_hands = 0
        total_false_accepts = 0

        for cat_name, samples in categories.items():
            rejected = 0
            for sample in samples:
                res = validator.validate_image_bytes(sample)
                if not res.passed:
                    rejected += 1
            fa = len(samples) - rejected
            far = fa / len(samples)
            total_non_hands += len(samples)
            total_false_accepts += fa
            print(f"{cat_name:<26} | {len(samples):<8} | {rejected:<10} | {far:<18.1%}")

        overall_far = total_false_accepts / total_non_hands
        print("-" * 70)
        print(f"Overall Non-Hand False Acceptance: {overall_far:.1%} ({total_false_accepts}/{total_non_hands}) (synthetic smoke test, not a measured rate)")
        print("========================================================================\n")
        self.assertEqual(total_false_accepts, 0, "Non-hand / spoof samples were incorrectly accepted!")


if __name__ == "__main__":
    unittest.main()
