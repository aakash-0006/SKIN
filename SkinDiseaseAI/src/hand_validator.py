"""Human Skin & Hand Input Validation, Anti-Spoofing, and Quality Pipeline.

Educational research prototype - NOT a medical device.
Validates that input images contain genuine human skin (across any body region: arms,
legs, torso, face, back, hands, or dermoscopic close-ups) with adequate skin coverage,
screens for presentation attacks (screens, prints), and checks image quality before
invoking the skin lesion classifier.
"""
from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from cvzone.HandTrackingModule import HandDetector
    _CVZONE_AVAILABLE = True
except Exception:
    _CVZONE_AVAILABLE = False


@dataclass
class ValidationResult:
    passed: bool
    status: str  # "accepted", "rejected", "uncertain"
    stage_failed: Optional[str] = None  # "skin_detection", "hand_detection", "spoof_check", "image_quality", "file_format"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    input_type: str = "Unknown"  # "Dermoscopy / Skin close-up", "Hand photo (unvalidated domain)", "Non-skin / Unrelated"
    skin_fraction: float = 0.0
    hand_count: int = 0
    hand_details: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    spoof_analysis: Dict[str, Any] = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None

    @property
    def anatomical_region(self) -> str:
        """Backwards-compatibility alias for input_type."""
        return self.input_type


class HumanSkinValidator:
    """Multi-stage validator enforcing:
    1. Skin tissue presence before disease classification.
    2. Optional hand-only landmark verification if strictly requested.
    3. Anti-spoofing heuristics (screen moiré, display borders, specular reflections).
    4. Focus, lighting, contrast, and resolution quality.

    Domain Notice: The downstream classifier was trained and evaluated strictly on
    HAM10000 and ISIC dermoscopic close-up images. Non-dermoscopic camera photos
    represent an unvalidated domain.
    """

    def __init__(
        self,
        min_skin_fraction: float = 0.15,
        min_detection_confidence: float = 0.55,
        min_blur_var: float = 30.0,
        min_luminance: float = 35.0,
        max_luminance: float = 235.0,
        min_contrast_std: float = 12.0,
        max_saturation_fraction: float = 0.28,
    ):
        self.min_skin_fraction = min_skin_fraction
        self.min_detection_confidence = min_detection_confidence
        self.min_blur_var = min_blur_var
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.min_contrast_std = min_contrast_std
        self.max_saturation_fraction = max_saturation_fraction

        self._hand_detector = None
        if _CVZONE_AVAILABLE:
            try:
                self._hand_detector = HandDetector(
                    staticMode=True,
                    maxHands=2,
                    detectionCon=min_detection_confidence,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize HandDetector: {e}")

    # ================= 1. Human Skin Detection =================
    def segment_human_skin(self, img_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute pixel mask and coverage fraction of skin tissue.

        Uses chromaticity across YCrCb, HSV, and normalized RGB to detect skin tissue
        while rejecting non-skin artifacts.
        """
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return 0.0, np.zeros((1, 1), dtype=np.uint8)

        # 1. YCrCb chromaticity (decouples luminance from skin chrominance)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        mask_ycrcb = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)

        # 2. HSV skin hue and saturation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        mask_hsv = ((h_channel <= 50) | (h_channel >= 165)) & (s_channel >= 25) & (v_channel >= 40)

        # 3. RGB physiological ordering (human skin has dominant red component)
        b, g, r = cv2.split(img_bgr)
        mask_rgb = (r > g) & (g >= b)

        combined_mask = mask_ycrcb & (mask_hsv | mask_rgb)

        # Clean up mask with small morphological opening to eliminate isolated noise spikes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(combined_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        skin_fraction = float((clean_mask > 0).mean())

        return skin_fraction, clean_mask

    def check_skin_region(self, img_bgr: np.ndarray) -> float:
        """Compute the fraction of human skin detected across any body region."""
        frac, _ = self.segment_human_skin(img_bgr)
        return frac

    # ================= 2. Hand Landmark Detection =================
    def detect_hands(self, img_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """Detect hands and extract landmarks using MediaPipe."""
        if self._hand_detector is None:
            return [], img_bgr

        img_copy = img_bgr.copy()
        try:
            hands, annotated = self._hand_detector.findHands(img_copy, draw=True, flipType=False)
            return hands, annotated
        except Exception as e:
            return [], img_bgr

    # ================= 3. Anti-Spoofing & Presentation Attack Check =================
    def check_spoof_and_authenticity(
        self,
        img_bgr: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Examine cues indicative of presentation attacks (screens, printed photos, replays).

        Scientific disclaimer:
        Heuristic frequency and edge analyses identify strong presentation artifacts (e.g. screen moiré,
        display bezels, high-frequency regular grids), but CANNOT guarantee that an image is genuine.
        Authenticity is marked as 'unverified_plausible' rather than unconditionally 'genuine'.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if roi_bbox is not None:
            bx, by, bw, bh = roi_bbox
            bx1, by1 = max(0, bx), max(0, by)
            bx2, by2 = min(w, bx + bw), min(h, by + bh)
            roi_gray = gray[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else gray
        else:
            roi_gray = gray

        # 1. 2D FFT Spectral Analysis (Moiré pattern detection)
        f = np.fft.fft2(cv2.resize(roi_gray, (256, 256)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)

        ch, cw = 128, 128
        y_idx, x_idx = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((x_idx - cw)**2 + (y_idx - ch)**2)
        high_freq_mask = (dist_from_center > 60) & (dist_from_center < 120)
        high_freq_energy = float(magnitude_spectrum[high_freq_mask].mean())
        center_energy = float(magnitude_spectrum[dist_from_center <= 30].mean())
        spectral_ratio = high_freq_energy / (center_energy + 1e-5)

        # 2. Rectilinear display border / photo frame detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=int(min(h, w) * 0.4), maxLineGap=10)
        has_screen_border = False
        num_border_lines = 0
        if lines is not None:
            for line in lines:
                coords = line.ravel()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    is_horiz = dy < 4 and dx > (0.35 * w)
                    is_vert = dx < 4 and dy > (0.35 * h)
                    if is_horiz or is_vert:
                        num_border_lines += 1
            if num_border_lines >= 2:
                has_screen_border = True

        # 3. Color Gamut and Specular Flash/Glare Check
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        extreme_specular_frac = float(((sat < 20) & (val > 245)).mean())

        suspicions = []
        if spectral_ratio > 0.85:
            suspicions.append("Periodic high-frequency pattern detected (possible digital screen moiré or print halftoning)")
        if has_screen_border:
            suspicions.append("Rectilinear display borders or physical photo frame detected")
        if extreme_specular_frac > 0.20:
            suspicions.append("Severe display/screen specular reflection detected")

        if suspicions:
            authenticity = "suspected_spoof"
            verdict_text = "Suspicious spoofing cues detected (potential screen, printed photo, or display replay)."
        else:
            authenticity = "unverified_plausible"
            verdict_text = "No obvious spoofing artifacts detected (presentation authenticity unverified without hardware depth sensor)."

        return {
            "authenticity": authenticity,
            "verdict_text": verdict_text,
            "suspicions": suspicions,
            "spectral_ratio": round(spectral_ratio, 3),
            "screen_border_detected": has_screen_border,
            "border_line_count": num_border_lines,
            "specular_fraction": round(extreme_specular_frac, 3),
            "disclaimer": "Anti-spoofing heuristics provide screening cues only and do not guarantee image genuineness.",
        }

    # ================= 4. Image Quality Verification =================
    def check_image_quality(self, img_bgr: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
        """Validate focus, illumination, contrast, and resolution."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(gray.mean())
        std_contrast = float(gray.std())
        dark_frac = float((gray < 20).mean())
        sat_frac = float((gray >= 245).mean())

        issues = []
        if min(h, w) < 180:
            issues.append(f"Image resolution is too low ({w}x{h} px, minimum 180x180 px required)")
        if blur_var < self.min_blur_var:
            issues.append(f"Image is too blurry or out of focus (sharpness score: {blur_var:.1f}, minimum: {self.min_blur_var:.1f})")
        if mean_lum < self.min_luminance or dark_frac > 0.40:
            issues.append(f"Image is severely underexposed or poorly lit (luminance: {mean_lum:.1f})")
        elif mean_lum > self.max_luminance or sat_frac > self.max_saturation_fraction:
            issues.append("Image is overexposed or has excessive specular glare")
        if std_contrast < self.min_contrast_std:
            issues.append("Image lacks sufficient contrast / detail")

        metrics = {
            "blur_variance": round(blur_var, 1),
            "luminance": round(mean_lum, 1),
            "contrast_std": round(std_contrast, 1),
            "dark_fraction": round(dark_frac, 3),
            "saturated_fraction": round(sat_frac, 3),
            "resolution": f"{w}x{h}",
        }
        return metrics, issues

    # ================= 5. Master Pipeline Verification =================
    def validate_image_bytes(
        self,
        image_bytes: bytes,
        mode: str = "all_skin",
    ) -> ValidationResult:
        """Run complete multi-stage input verification:

        Parameters:
        - mode:
          - "all_skin" (Default): Accepts skin anywhere on the human body (arms, legs, torso, face, hands, dermoscopy).
          - "hand_only": Strictly requires a detected human hand with landmarks.

        Pipeline Stages:
        1. File decoding and dimensions
        2. Human skin coverage & anatomical landmark detection
        3. Anti-spoof presentation attack check
        4. Image quality (blur, lighting, contrast, resolution)
        """
        # Step 0: Read and decode bytes
        if not image_bytes:
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                reasons=["The submitted file is empty."],
            )

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.load()
        except (UnidentifiedImageError, OSError, ValueError):
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                reasons=["The submitted file is not a valid JPEG or PNG image."],
            )

        img_rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Step 1: Detect hands if visible
        hands, annotated = self.detect_hands(img_bgr)
        hand_count = len(hands)

        # Check for multiple hands if hand detection fired
        if hand_count > 1:
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="hand_detection",
                reasons=[f"Multiple hands detected ({hand_count} hands found). Please present a single target region for clear analysis."],
                hand_count=hand_count,
                annotated_image=annotated,
            )

        # Step 2: Human Skin Detection
        skin_frac, skin_mask = self.segment_human_skin(img_bgr)

        # Determine input type classification
        if hand_count == 1:
            input_type = "Hand photo (unvalidated domain)"
            display_annotated = annotated
            hand_info = [{
                "type": hands[0].get("type", "Hand"),
                "score": round(float(hands[0].get("score", [1.0])[0]) if isinstance(hands[0].get("score"), (list, tuple)) else 1.0, 3),
                "bbox": hands[0].get("bbox"),
                "landmarks_count": len(hands[0].get("lmList", [])),
            }]
        elif skin_frac >= 0.15:
            input_type = "Skin-like pixels (colour check only)"
            # Draw semi-transparent contour overlay highlighting detected skin area
            display_annotated = img_bgr.copy()
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_annotated, contours, -1, (0, 220, 0), 2)
            hand_info = []
        else:
            input_type = "Non-skin / Unrelated content"
            display_annotated = img_bgr
            hand_info = []

        # Check mode requirements
        if mode == "hand_only" and hand_count == 0:
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="hand_detection",
                reasons=["No hand detected. (Hand-only mode is active). Please upload an image of a human hand."],
                input_type=input_type,
                hand_count=0,
                skin_fraction=round(skin_frac, 3),
                annotated_image=img_bgr,
            )

        # Verify minimum skin coverage (if no hand was detected)
        if skin_frac < self.min_skin_fraction and hand_count == 0:
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="skin_detection",
                reasons=[
                    f"No skin tissue detected (skin coverage: {skin_frac:.1%}, minimum {self.min_skin_fraction:.0%} required). "
                    "Please upload a clear dermoscopic or skin lesion photograph."
                ],
                input_type=input_type,
                skin_fraction=round(skin_frac, 3),
                hand_count=hand_count,
                annotated_image=display_annotated,
            )

        # Step 3: Fake Image / Anti-Spoofing Check
        roi_bbox = hands[0].get("bbox") if hand_count == 1 else None
        spoof_res = self.check_spoof_and_authenticity(img_bgr, roi_bbox=roi_bbox)
        if spoof_res["authenticity"] == "suspected_spoof":
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="spoof_check",
                reasons=["Suspicious presentation attack / spoof detected: " + "; ".join(spoof_res["suspicions"])],
                input_type=input_type,
                hand_count=hand_count,
                hand_details=hand_info,
                skin_fraction=round(skin_frac, 3),
                spoof_analysis=spoof_res,
                annotated_image=display_annotated,
            )

        # Step 4: Image Quality Check
        quality_metrics, quality_issues = self.check_image_quality(img_bgr)
        if quality_issues:
            return ValidationResult(
                passed=False,
                status="rejected",
                stage_failed="image_quality",
                reasons=quality_issues,
                input_type=input_type,
                hand_count=hand_count,
                hand_details=hand_info,
                skin_fraction=round(skin_frac, 3),
                quality_metrics=quality_metrics,
                spoof_analysis=spoof_res,
                annotated_image=display_annotated,
            )

        warnings = []
        if spoof_res["authenticity"] == "unverified_plausible":
            warnings.append("Authenticity Notice: Anti-spoofing heuristics found no obvious screen/print artifacts, but physical presentation cannot be guaranteed without dedicated hardware.")
        if hand_count == 1:
            warnings.append("Domain notice: Hand photo detected. The classifier was trained and evaluated strictly on dermoscopic close-ups (HAM10000/ISIC); clinical macro-photos represent an unvalidated domain.")
        else:
            warnings.append("Domain notice: Classifier is validated on dermoscopic close-up images (HAM10000/ISIC). Non-dermoscopic camera photos represent an unvalidated domain.")
        warnings.append("Educational Prototype: Validating skin presence does NOT certify that the skin area is healthy or substitute for a dermatologist.")

        # Success! All checks passed across human skin region
        return ValidationResult(
            passed=True,
            status="accepted",
            stage_failed=None,
            reasons=[],
            warnings=warnings,
            input_type=input_type,
            hand_count=hand_count,
            hand_details=hand_info,
            skin_fraction=round(skin_frac, 3),
            quality_metrics=quality_metrics,
            spoof_analysis=spoof_res,
            annotated_image=display_annotated,
        )


# Backwards compatibility alias
HandValidator = HumanSkinValidator

_GLOBAL_VALIDATOR = None

def get_hand_validator() -> HumanSkinValidator:
    global _GLOBAL_VALIDATOR
    if _GLOBAL_VALIDATOR is None:
        _GLOBAL_VALIDATOR = HumanSkinValidator()
    return _GLOBAL_VALIDATOR

get_skin_validator = get_hand_validator
