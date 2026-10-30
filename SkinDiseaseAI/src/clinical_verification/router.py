"""Stage 3 — Image Type and Anatomical Routing.

Educational research prototype — NOT a medical diagnostic device.
Determines whether an input is:
- Dermoscopic close-up (eligible for the 7-class HAM10000/ISIC classifier).
- Hand photograph (MediaPipe localized hand; routed to hand appearance analysis).
- Ordinary phone-camera skin photograph (routed to Stage R clinical appearance model).
- Unsupported or multi-region image (multiple hands, non-skin, or out-of-distribution).

Strict domain boundary:
Ordinary camera photos of hands, arms, legs, or face are NEVER sent to the
7-class dermoscopic classifier (HAM10000). Separate routes protect clinical safety.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from cvzone.HandTrackingModule import HandDetector
    _CVZONE_AVAILABLE = True
except Exception:
    _CVZONE_AVAILABLE = False


@dataclass
class RoutingResult:
    route: str  # "dermoscopic_lesion", "hand_photo", "clinical_macro_skin", "unsupported"
    route_name: str
    target_pipeline: str
    eligible_for_7class_lesion: bool = False
    eligible_for_clinical_appearance: bool = False
    eligible_for_hand_appearance: bool = False
    hand_count: int = 0
    hand_details: List[Dict[str, Any]] = field(default_factory=list)
    hand_crop_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    gate_verdict: Optional[Dict[str, Any]] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None


class AnatomicalRouter:
    """Stage 3 Anatomical and Image Type Router."""

    def __init__(self, min_hand_confidence: float = 0.55):
        self.min_hand_confidence = min_hand_confidence
        self._hand_detector = None
        if _CVZONE_AVAILABLE:
            try:
                self._hand_detector = HandDetector(
                    staticMode=True,
                    maxHands=2,
                    detectionCon=min_hand_confidence,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize HandDetector: {e}")

    def detect_hands(self, img_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect hands and extract landmark coordinates."""
        if self._hand_detector is None:
            return [], img_bgr

        img_copy = img_bgr.copy()
        try:
            hands, annotated = self._hand_detector.findHands(img_copy, draw=True, flipType=False)
            return hands, annotated
        except Exception:
            return [], img_bgr

    def route_image(
        self,
        img_bgr: np.ndarray,
        data_bytes: bytes,
    ) -> RoutingResult:
        """Route input based on anatomical landmarks and dermoscopic Mahalanobis gate."""
        from src.gate import check_image

        # 1. Check for Human Hand Landmarks
        hands, annotated_hands = self.detect_hands(img_bgr)
        hand_count = len(hands)

        if hand_count > 1:
            return RoutingResult(
                route="unsupported",
                route_name="Multiple hands detected",
                target_pipeline="None (Blocked)",
                eligible_for_7class_lesion=False,
                eligible_for_clinical_appearance=False,
                eligible_for_hand_appearance=False,
                hand_count=hand_count,
                reasons=[f"Multiple hands detected ({hand_count} hands found). Please present a single target region."],
                annotated_image=annotated_hands,
            )

        if hand_count == 1:
            h_info = hands[0]
            bbox = h_info.get("bbox")  # (x, y, w, h)
            hand_details = [{
                "type": h_info.get("type", "Hand"),
                "score": round(float(h_info.get("score", [1.0])[0]) if isinstance(h_info.get("score"), (list, tuple)) else 1.0, 3),
                "bbox": bbox,
                "landmarks_count": len(h_info.get("lmList", [])),
            }]
            warnings = [
                "Domain Boundary Notice: Hand photo detected. The 7-class lesion classifier (HAM10000) is strictly suppressed. "
                "The image is routed exclusively to the Hand Appearance Analysis pipeline."
            ]
            return RoutingResult(
                route="hand_photo",
                route_name="Hand photograph (macro camera)",
                target_pipeline="Clinical Hand Appearance Model (Stage R & Normal Hand)",
                eligible_for_7class_lesion=False,
                eligible_for_clinical_appearance=True,
                eligible_for_hand_appearance=True,
                hand_count=1,
                hand_details=hand_details,
                hand_crop_bbox=bbox,
                warnings=warnings,
                annotated_image=annotated_hands,
            )

        # 2. No Hand Detected: Evaluate Dermoscopic vs Clinical Macro Skin
        gate_verdict = check_image(data_bytes)
        gate_status = gate_verdict.get("status")

        if gate_status == "accepted":
            # Genuine Dermoscopy Domain: Magnified pigmented lesion close-up
            return RoutingResult(
                route="dermoscopic_lesion",
                route_name="Dermoscopic lesion close-up (validated domain)",
                target_pipeline="7-Class Dermoscopic Classifier (HAM10000 / ISIC)",
                eligible_for_7class_lesion=True,
                eligible_for_clinical_appearance=False,
                eligible_for_hand_appearance=False,
                hand_count=0,
                gate_verdict=gate_verdict,
                warnings=[
                    "Dermoscopy domain confirmed: Image matches optical characteristics of dermoscopic training data (HAM10000). "
                    "Educational research prototype — not a medical diagnosis."
                ],
                annotated_image=img_bgr,
            )
        else:
            # Clinical Macro Photo of Non-Hand Skin (limbs, torso, face, neck)
            reasons_gate = gate_verdict.get("reasons", [])
            warnings = [
                "Domain Boundary Notice: This is an ordinary phone-camera skin photograph, NOT a dermoscopic close-up. "
                "The 7-class lesion classifier (HAM10000) is strictly suppressed to prevent invalid out-of-domain predictions. "
                "The image is routed to the Clinical Skin Appearance Model."
            ]
            return RoutingResult(
                route="clinical_macro_skin",
                route_name="Ordinary phone-camera skin photograph (macro clinical domain)",
                target_pipeline="Clinical Skin Appearance Model (Stage R)",
                eligible_for_7class_lesion=False,
                eligible_for_clinical_appearance=True,
                eligible_for_hand_appearance=False,
                hand_count=0,
                gate_verdict=gate_verdict,
                warnings=warnings,
                annotated_image=img_bgr,
            )
