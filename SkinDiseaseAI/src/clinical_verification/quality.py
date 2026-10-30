"""Stage 1 — Image Quality Verification.

Educational research prototype — NOT a medical diagnostic device.
Evaluates file integrity, dimensions, format, focus (blur), illumination, and contrast.
Rejects corrupted, unsupported, out-of-focus, or poorly illuminated images.
"""
from dataclasses import dataclass, field
import io
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MIN_SIDE = 180
MAX_BYTES = 15 * 1024 * 1024
SUPPORTED_FORMATS = ("JPEG", "PNG")

DEFAULT_MIN_BLUR = 30.0
DEFAULT_MIN_LUM = 35.0
DEFAULT_MAX_LUM = 235.0
DEFAULT_MIN_CONTRAST = 12.0
DEFAULT_MAX_DARK_FRAC = 0.40
DEFAULT_MAX_SAT_FRAC = 0.28


@dataclass
class QualityCheckResult:
    passed: bool
    status: str  # "accepted" or "rejected"
    stage_failed: Optional[str] = None  # "file_format", "image_dimensions", "image_quality"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    format: Optional[str] = None
    width: int = 0
    height: int = 0
    pil_image: Optional[Image.Image] = None
    img_bgr: Optional[np.ndarray] = None


class ImageQualityChecker:
    """Stage 1 Image Quality & Integrity Checker."""

    def __init__(
        self,
        min_blur: float = DEFAULT_MIN_BLUR,
        min_lum: float = DEFAULT_MIN_LUM,
        max_lum: float = DEFAULT_MAX_LUM,
        min_contrast: float = DEFAULT_MIN_CONTRAST,
        max_dark_frac: float = DEFAULT_MAX_DARK_FRAC,
        max_sat_frac: float = DEFAULT_MAX_SAT_FRAC,
        min_side: int = MIN_SIDE,
        max_bytes: int = MAX_BYTES,
    ):
        self.min_blur = min_blur
        self.min_lum = min_lum
        self.max_lum = max_lum
        self.min_contrast = min_contrast
        self.max_dark_frac = max_dark_frac
        self.max_sat_frac = max_sat_frac
        self.min_side = min_side
        self.max_bytes = max_bytes

    def check(self, data: bytes) -> QualityCheckResult:
        """Run complete Stage 1 quality evaluation on raw image bytes."""
        # 1. Byte existence and file size limits
        if not data or len(data) == 0:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                reasons=["The submitted file is empty (0 bytes)."],
            )

        if len(data) > self.max_bytes:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                reasons=[f"File size exceeds limit ({len(data)/(1024*1024):.1f} MB, maximum {self.max_bytes/(1024*1024):.0f} MB)."],
            )

        # 2. File integrity & decoding
        try:
            im_probe = Image.open(io.BytesIO(data))
            im_probe.verify()
            fmt = im_probe.format
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                reasons=[f"Corrupted or unreadable image file: {str(e)}"],
            )

        if fmt not in SUPPORTED_FORMATS:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                format=fmt,
                reasons=[f"Unsupported image format '{fmt}'. Only JPG and PNG are supported."],
            )

        try:
            pil_img = Image.open(io.BytesIO(data))
            pil_img.load()
            pil_rgb = pil_img.convert("RGB")
        except Exception as e:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="file_format",
                format=fmt,
                reasons=[f"Could not decode image pixels: {str(e)}"],
            )

        w, h = pil_rgb.size

        # 3. Dimensional limits
        if min(w, h) < self.min_side:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="image_dimensions",
                format=fmt,
                width=w,
                height=h,
                reasons=[f"Image resolution too low ({w}x{h} px). Minimum dimension required is {self.min_side}x{self.min_side} px."],
                pil_image=pil_rgb,
            )

        # Convert to BGR array for OpenCV analysis
        img_rgb = np.asarray(pil_rgb, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 4. Blur / Sharpness calculation
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 5. Lighting, Contrast & Glare calculations
        mean_lum = float(gray.mean())
        std_contrast = float(gray.std())
        dark_frac = float((gray < 20).mean())
        sat_frac = float((gray >= 245).mean())

        metrics = {
            "blur_variance": round(blur_var, 2),
            "luminance_mean": round(mean_lum, 2),
            "contrast_std": round(std_contrast, 2),
            "dark_fraction": round(dark_frac, 4),
            "saturated_fraction": round(sat_frac, 4),
            "width": w,
            "height": h,
        }

        issues = []
        if blur_var < self.min_blur:
            issues.append(f"Image is out of focus or severely blurry (sharpness variance: {blur_var:.1f}, minimum: {self.min_blur:.1f}).")
        if mean_lum < self.min_lum or dark_frac > self.max_dark_frac:
            issues.append(f"Image is underexposed or poorly illuminated (luminance: {mean_lum:.1f}, dark pixels: {dark_frac:.1%}).")
        elif mean_lum > self.max_lum or sat_frac > self.max_sat_frac:
            issues.append(f"Image is overexposed or has excessive specular glare (luminance: {mean_lum:.1f}, saturated pixels: {sat_frac:.1%}).")
        if std_contrast < self.min_contrast:
            issues.append(f"Image lacks sufficient contrast / detail (contrast std: {std_contrast:.1f}, minimum: {self.min_contrast:.1f}).")

        warnings = []
        if 30.0 <= blur_var < 50.0:
            warnings.append(f"Moderate focus blur detected ({blur_var:.1f}). Clearer results require holding the camera steady.")
        if 0.15 <= sat_frac <= self.max_sat_frac:
            warnings.append(f"Noticeable specular reflection or flash glare present ({sat_frac:.1%}).")

        if issues:
            return QualityCheckResult(
                passed=False,
                status="rejected",
                stage_failed="image_quality",
                format=fmt,
                width=w,
                height=h,
                reasons=issues,
                warnings=warnings,
                metrics=metrics,
                pil_image=pil_rgb,
                img_bgr=img_bgr,
            )

        return QualityCheckResult(
            passed=True,
            status="accepted",
            stage_failed=None,
            format=fmt,
            width=w,
            height=h,
            reasons=[],
            warnings=warnings,
            metrics=metrics,
            pil_image=pil_rgb,
            img_bgr=img_bgr,
        )
