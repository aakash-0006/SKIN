"""Hand detection, landmark localization, and bounding-box cropping utilities.

Extracts tight, padded hand crops to prevent convolutional models from latching onto
surrounding white background boards or studio artifacts.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

try:
    from cvzone.HandTrackingModule import HandDetector
    _CVZONE_AVAILABLE = True
except Exception:
    _CVZONE_AVAILABLE = False


@dataclass
class CropResult:
    success: bool
    cropped_bgr: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    hand_count: int = 0
    failure_reason: Optional[str] = None
    original_shape: Optional[Tuple[int, int]] = None
    crop_shape: Optional[Tuple[int, int]] = None


class HandCropper:
    """Detects human hands and crops the region-of-interest with a safety margin."""

    def __init__(
        self,
        detection_confidence: float = 0.50,
        pad_fraction: float = 0.10,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.pad_fraction = pad_fraction
        self.target_size = target_size
        self.detection_confidence = detection_confidence
        self.detector = None

        if _CVZONE_AVAILABLE:
            try:
                self.detector = HandDetector(
                    staticMode=True,
                    maxHands=2,
                    detectionCon=detection_confidence,
                )
            except Exception as e:
                print(f"Warning: HandDetector initialization failed: {e}")

    def crop_hand(self, img_bgr: np.ndarray) -> CropResult:
        """Localize and crop the primary hand from an input image.

        Args:
            img_bgr: Input BGR image.

        Returns:
            CropResult containing cropped BGR array or failure diagnostic.
        """
        if img_bgr is None or img_bgr.size == 0:
            return CropResult(success=False, failure_reason="EMPTY_OR_INVALID_IMAGE")

        h, w = img_bgr.shape[:2]
        if h < 50 or w < 50:
            return CropResult(success=False, failure_reason="RESOLUTION_TOO_LOW")

        if self.detector is None:
            return CropResult(success=False, failure_reason="DETECTOR_UNAVAILABLE")

        try:
            hands, _ = self.detector.findHands(img_bgr, draw=False)
        except Exception as e:
            return CropResult(success=False, failure_reason=f"DETECTION_ERROR: {str(e)}")

        if not hands:
            return CropResult(
                success=False,
                hand_count=0,
                failure_reason="NO_HAND_DETECTED",
                original_shape=(h, w),
            )

        # Select primary hand (largest bbox area)
        primary_hand = max(hands, key=lambda d: d["bbox"][2] * d["bbox"][3])
        bx, by, bw, bh = primary_hand["bbox"]

        # Validate bounding box
        if bw < 20 or bh < 20:
            return CropResult(
                success=False,
                hand_count=len(hands),
                failure_reason="DEGENERATE_BOUNDING_BOX",
                original_shape=(h, w),
            )

        # Calculate padded crop with boundary clamping
        pad_x = int(bw * self.pad_fraction)
        pad_y = int(bh * self.pad_fraction)

        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return CropResult(
                success=False,
                hand_count=len(hands),
                failure_reason="DEGENERATE_CROP_DIMENSIONS",
                original_shape=(h, w),
            )

        crop_raw = img_bgr[y1:y2, x1:x2]
        crop_h, crop_w = crop_raw.shape[:2]

        # Resize to standard target size for neural feature extraction
        crop_resized = cv2.resize(
            crop_raw,
            self.target_size,
            interpolation=cv2.INTER_AREA if (crop_h > self.target_size[1]) else cv2.INTER_LINEAR,
        )

        return CropResult(
            success=True,
            cropped_bgr=crop_resized,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            hand_count=len(hands),
            original_shape=(h, w),
            crop_shape=(crop_h, crop_w),
        )
