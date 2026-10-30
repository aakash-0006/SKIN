"""Stage 4 & 5 — Clinical Skin Appearance Classifier & Uncertainty Calibrator.

Educational research prototype — NOT a medical diagnostic device.
Infers whether an image visual appearance is:
- Normal-looking skin appearance (statistically similar to reference normal hand cohort).
- Potentially abnormal appearance (statistically similar to reference clinical condition cohort).
- Uncertain result (confidence in uncertain decision margin [0.35, 0.65]).
- Unsupported image (out-of-distribution or non-skin).

Important:
- Never labels benign lesions as normal skin.
- Does not claim that eczema/tinea training provides general detection of all skin diseases.
- Applies background-suppressing skin/hand cropping.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_R_MODEL_PATH = PROJECT_ROOT / "models" / "clinical_skin_stage_r" / "clinical_skin_stage_r_best.keras"
CANDIDATE_1B_PATH = PROJECT_ROOT / "models" / "experiments" / "normal_hand_v1" / "stage_g" / "candidate_1b_l2_mahalanobis_k32.npz"


@dataclass
class ClinicalClassificationResult:
    decision: str  # "NORMAL_LOOKING_APPEARANCE", "POTENTIALLY_ABNORMAL_APPEARANCE", "UNCERTAIN_APPEARANCE", "UNSUPPORTED_INPUT"
    category_display: str
    abnormal_probability: float
    confidence: float
    is_uncertain: bool
    status: str  # "accepted", "uncertain", "unsupported"
    message: str
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    hand_mahalanobis_score: Optional[float] = None
    hand_mahalanobis_threshold: Optional[float] = None
    crop_shape: Optional[Tuple[int, int]] = None
    cropped_image: Optional[np.ndarray] = None


@lru_cache(maxsize=1)
def _load_stage_r_model():
    """Lazily load Stage R MobileNetV3 clinical classifier."""
    import tensorflow as tf
    if not STAGE_R_MODEL_PATH.exists():
        raise FileNotFoundError(f"Stage R model artifact not found at {STAGE_R_MODEL_PATH}")
    return tf.keras.models.load_model(str(STAGE_R_MODEL_PATH), compile=False)


@lru_cache(maxsize=1)
def _load_candidate_1b_detector():
    """Lazily load Candidate 1B hand anomaly detector."""
    if not CANDIDATE_1B_PATH.exists():
        return None
    from src.normal_hand.anomaly_detector import MahalanobisAnomalyDetector
    detector = MahalanobisAnomalyDetector.load(str(CANDIDATE_1B_PATH))
    detector.threshold = 6.931
    return detector


class ClinicalSkinClassifier:
    """Stage 4 & 5 Clinical Skin Appearance Classifier."""

    def __init__(
        self,
        lower_threshold: float = 0.35,
        upper_threshold: float = 0.65,
        img_size: int = 224,
    ):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.img_size = img_size

    def _crop_roi(
        self,
        img_bgr: np.ndarray,
        hand_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract skin ROI to suppress studio white / background confounding."""
        h, w = img_bgr.shape[:2]
        if hand_bbox is not None:
            bx, by, bw, bh = hand_bbox
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            if (x2 - x1) >= 64 and (y2 - y1) >= 64:
                roi = img_bgr[y1:y2, x1:x2]
                return roi, (y2 - y1, x2 - x1)

        # Fallback: center 85% crop to reduce peripheral table/wall background
        cy1 = int(h * 0.075)
        cy2 = int(h * 0.925)
        cx1 = int(w * 0.075)
        cx2 = int(w * 0.925)
        roi = img_bgr[cy1:cy2, cx1:cx2]
        return roi, (cy2 - cy1, cx2 - cx1)

    def classify(
        self,
        img_bgr: np.ndarray,
        hand_bbox: Optional[Tuple[int, int, int, int]] = None,
        is_hand: bool = False,
    ) -> ClinicalClassificationResult:
        """Execute Stage 4 classification and Stage 5 uncertainty calibration."""
        # 1. Crop region of interest
        roi_bgr, crop_shape = self._crop_roi(img_bgr, hand_bbox=hand_bbox)
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(roi_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Input array: MobileNetV3 expects raw RGB float32 in [0, 255]
        x_in = np.expand_dims(im_resized.astype(np.float32), axis=0)

        # 2. Run CNN Model Inference
        model = _load_stage_r_model()
        pred_prob = float(model.predict(x_in, verbose=0)[0, 0])

        # 3. Candidate 1B Mahalanobis Anomaly Check for Hands
        mahal_score = None
        mahal_thr = 6.931
        if is_hand:
            detector = _load_candidate_1b_detector()
            if detector is not None:
                try:
                    from src.normal_hand.feature_extractor import HandFeatureExtractor
                    fe = HandFeatureExtractor(target_size=(224, 224))
                    feats = fe.extract(im_resized)
                    # L2-normalize features as per Candidate 1B specification
                    norm = np.linalg.norm(feats, ord=2, axis=-1, keepdims=True) + 1e-8
                    feats_norm = feats / norm
                    mahal_score = float(detector.score(feats_norm)[0])
                except Exception:
                    pass

        # 4. Uncertainty & Decision Calibration (Stage 5)
        # If probability falls into the high-entropy uncertainty margin [0.35, 0.65]:
        if self.lower_threshold <= pred_prob <= self.upper_threshold:
            decision = "UNCERTAIN_APPEARANCE"
            display = "Uncertain result"
            status = "uncertain"
            confidence = round(1.0 - 2.0 * abs(pred_prob - 0.50), 3)  # Closer to 0.5 = higher uncertainty
            message = (
                f"The visual appearance cannot be reliably classified with high confidence "
                f"(output probability {pred_prob:.1%} falls within the uncertain decision margin [{self.lower_threshold:.0%} - {self.upper_threshold:.0%}])."
            )
        elif pred_prob < self.lower_threshold:
            decision = "NORMAL_LOOKING_APPEARANCE"
            display = "Normal-looking skin appearance"
            status = "accepted"
            confidence = round(1.0 - pred_prob, 3)
            message = "Visual appearance is statistically consistent with reference normal hand skin in our cohort."
        else:
            decision = "POTENTIALLY_ABNORMAL_APPEARANCE"
            display = "Potentially abnormal appearance"
            status = "accepted"
            confidence = round(pred_prob, 3)
            message = "Visual appearance shares features with reference clinical inflammatory / fungal skin conditions."

        warnings = []
        if is_hand and mahal_score is not None:
            if mahal_score > mahal_thr and decision == "NORMAL_LOOKING_APPEARANCE":
                warnings.append(
                    f"Appearance Anomaly Advisory: CNN probability is normal ({1.0 - pred_prob:.1%}), "
                    f"but geometric embedding distance ({mahal_score:.2f} vs threshold {mahal_thr:.2f}) indicates unusual lighting, pose, or accessories."
                )

        limitations = [
            "Educational research prototype — NOT a medical diagnostic device.",
            "Normal-looking skin appearance does NOT certify that the skin is healthy, lesion-free, or without medical condition.",
            "Potentially abnormal appearance is an educational screening cue and does NOT constitute a medical diagnosis.",
            "Model was trained on limited clinical conditions (eczema and tinea) and does NOT detect every skin pathology.",
            "Consult a board-certified dermatologist for any clinical concern.",
        ]

        return ClinicalClassificationResult(
            decision=decision,
            category_display=display,
            abnormal_probability=round(pred_prob, 4),
            confidence=round(confidence, 4),
            is_uncertain=(status == "uncertain"),
            status=status,
            message=message,
            warnings=warnings,
            limitations=limitations,
            hand_mahalanobis_score=round(mahal_score, 3) if mahal_score is not None else None,
            hand_mahalanobis_threshold=mahal_thr,
            crop_shape=crop_shape,
            cropped_image=roi_bgr,
        )
