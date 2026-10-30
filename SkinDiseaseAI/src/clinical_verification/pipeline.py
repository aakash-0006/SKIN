"""Unified Clinical Verification and Classification Pipeline.

Educational research prototype — NOT a medical diagnostic device.
Orchestrates:
- Stage 1: Image Quality Verification
- Stage 2: Skin and Image Domain Verification
- Stage 3: Anatomical and Modality Routing
- Stage 4: Clinical Appearance Classification (Normal vs Potentially Abnormal)
- Stage 5: Uncertainty Calibration and Out-of-Distribution Handling
- Stage 6: Preserved 7-Class Dermoscopic Classifier Integration (Dermoscopy Only)
"""
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .quality import ImageQualityChecker, QualityCheckResult
from .domain_verifier import SkinDomainVerifier, DomainVerificationResult
from .router import AnatomicalRouter, RoutingResult
from .clinical_classifier import ClinicalSkinClassifier, ClinicalClassificationResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EDUCATIONAL_DISCLAIMER = (
    "Educational research prototype — NOT a medical diagnostic device. "
    "This system provides educational classification cues across clinical photographs and dermoscopic images. "
    "It cannot determine whether skin is healthy or diagnose any clinical condition. "
    "Consult a qualified dermatologist or clinician for any medical concern."
)


@dataclass
class PipelineOutput:
    ok: bool
    status: str  # "accepted", "rejected", "uncertain", "unsupported", "low_quality"
    category: str
    decision_code: str
    route: str  # "dermoscopic_lesion", "hand_photo", "clinical_macro_skin", "unsupported"
    route_name: str
    target_pipeline: str
    confidence: float
    message: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scores: List[Tuple[str, str, float]] = field(default_factory=list)  # (code, name, score) for 7-class
    melanoma_flag: bool = False
    melanoma_flag_info: Optional[Dict[str, Any]] = None
    clinical_result: Optional[ClinicalClassificationResult] = None
    quality_result: Optional[QualityCheckResult] = None
    domain_result: Optional[DomainVerificationResult] = None
    routing_result: Optional[RoutingResult] = None
    annotated_image: Optional[np.ndarray] = None
    cropped_image: Optional[np.ndarray] = None
    disclaimer: str = EDUCATIONAL_DISCLAIMER


class ClinicalVerificationPipeline:
    """Master orchestrator enforcing end-to-end multi-stage clinical verification."""

    def __init__(
        self,
        quality_checker: Optional[ImageQualityChecker] = None,
        domain_verifier: Optional[SkinDomainVerifier] = None,
        router: Optional[AnatomicalRouter] = None,
        clinical_classifier: Optional[ClinicalSkinClassifier] = None,
    ):
        self.quality_checker = quality_checker or ImageQualityChecker()
        self.domain_verifier = domain_verifier or SkinDomainVerifier()
        self.router = router or AnatomicalRouter()
        self.clinical_classifier = clinical_classifier or ClinicalSkinClassifier()

    def process(
        self,
        image_bytes: bytes,
        dermoscopy_model_path: Optional[str] = None,
        bypass_quality: bool = False,
    ) -> PipelineOutput:
        """Execute full verification pipeline on raw input bytes."""
        # ================= STAGE 1: IMAGE QUALITY VERIFICATION =================
        q_res = self.quality_checker.check(image_bytes)
        if not q_res.passed and not bypass_quality:
            return PipelineOutput(
                ok=False,
                status="low_quality",
                category="Low image quality",
                decision_code="REJECTED_QUALITY",
                route="unsupported",
                route_name="Quality rejection",
                target_pipeline="None (Blocked)",
                confidence=0.0,
                message=f"Image rejected during quality verification: {q_res.reasons[0]}",
                reasons=q_res.reasons,
                warnings=q_res.warnings,
                quality_result=q_res,
                annotated_image=q_res.img_bgr,
            )

        pil_rgb = q_res.pil_image
        img_bgr = q_res.img_bgr

        # Quick check for hand landmarks to inform domain verifier
        hands_probe, _ = self.router.detect_hands(img_bgr)
        is_hand_present = len(hands_probe) == 1
        hand_bbox_probe = hands_probe[0].get("bbox") if is_hand_present else None

        # ================= STAGE 2: SKIN & DOMAIN VERIFICATION =================
        d_res = self.domain_verifier.verify(
            img_bgr,
            pil_rgb,
            is_hand_detected=is_hand_present,
            roi_bbox=hand_bbox_probe,
        )
        if not d_res.passed:
            cat_display = (
                "Non-skin image" if d_res.stage_failed == "non_skin_object"
                else "Artificial or drawing / anime" if d_res.stage_failed == "drawing_or_anime"
                else "Unsupported image (insufficient skin)" if d_res.stage_failed == "skin_coverage"
                else "Suspicious presentation attack / spoof"
            )
            return PipelineOutput(
                ok=False,
                status="rejected",
                category=cat_display,
                decision_code=f"REJECTED_{d_res.stage_failed.upper()}",
                route="unsupported",
                route_name="Domain gate rejection",
                target_pipeline="None (Blocked)",
                confidence=0.0,
                message=f"Image rejected during domain verification: {d_res.reasons[0]}",
                reasons=d_res.reasons,
                warnings=d_res.warnings,
                quality_result=q_res,
                domain_result=d_res,
                annotated_image=d_res.annotated_image,
            )

        # ================= STAGE 3: ANATOMICAL & MODALITY ROUTING =================
        r_res = self.router.route_image(img_bgr, image_bytes)
        if r_res.route == "unsupported":
            return PipelineOutput(
                ok=False,
                status="unsupported",
                category="Unsupported image",
                decision_code="REJECTED_MULTI_HAND",
                route=r_res.route,
                route_name=r_res.route_name,
                target_pipeline=r_res.target_pipeline,
                confidence=0.0,
                message=r_res.reasons[0],
                reasons=r_res.reasons,
                warnings=r_res.warnings,
                quality_result=q_res,
                domain_result=d_res,
                routing_result=r_res,
                annotated_image=r_res.annotated_image,
            )

        # Merge warnings from stages 1-3
        pipeline_warnings = list(q_res.warnings) + list(d_res.warnings) + list(r_res.warnings)

        # ================= ROUTE A: DERMOSCOPIC LESION ROUTE =================
        if r_res.route == "dermoscopic_lesion":
            from src.predict import NAMES, EXPLAIN, load_artifacts

            try:
                model, idx_to_class, img_size = load_artifacts(dermoscopy_model_path)
            except Exception as e:
                return PipelineOutput(
                    ok=False,
                    status="unsupported",
                    category="Dermoscopic classifier unavailable",
                    decision_code="MODEL_UNAVAILABLE",
                    route=r_res.route,
                    route_name=r_res.route_name,
                    target_pipeline=r_res.target_pipeline,
                    confidence=0.0,
                    message=f"Dermoscopic model could not be loaded: {str(e)}",
                    reasons=[f"Model loading error: {str(e)}"],
                    warnings=pipeline_warnings,
                    quality_result=q_res,
                    domain_result=d_res,
                    routing_result=r_res,
                )

            # Preprocess for EfficientNetB0 dermoscopic model
            im_resized = pil_rgb.resize((img_size, img_size), Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.array(im_resized, dtype=np.float32), 0)
            preds = model.predict(arr, verbose=0)[0]

            scores = []
            for idx, prob in enumerate(preds):
                c_code = idx_to_class[idx]
                c_name = NAMES.get(c_code, c_code.upper())
                scores.append((c_code, c_name, float(prob)))
            scores.sort(key=lambda x: x[2], reverse=True)

            top_code, top_name, top_score = scores[0]

            # Melanoma follow-up screening flag check
            mel_score = next((s[2] for s in scores if s[0] == "mel"), 0.0)
            mel_thr = 0.347
            mel_flag = mel_score >= mel_thr
            mel_info = {
                "mel_prob": mel_score,
                "threshold": mel_thr,
            } if mel_flag else None

            # Abstain rule if top score < 40%
            if top_score < 0.40:
                return PipelineOutput(
                    ok=True,
                    status="uncertain",
                    category="Uncertain result (low top score)",
                    decision_code="UNCERTAIN_LOW_CONFIDENCE",
                    route=r_res.route,
                    route_name=r_res.route_name,
                    target_pipeline=r_res.target_pipeline,
                    confidence=round(top_score, 4),
                    message=f"Lesion appearance is diffuse or ambiguous (top score {top_score:.1%} is below 40% threshold).",
                    warnings=pipeline_warnings,
                    scores=scores,
                    melanoma_flag=mel_flag,
                    melanoma_flag_info=mel_info,
                    quality_result=q_res,
                    domain_result=d_res,
                    routing_result=r_res,
                    annotated_image=img_bgr,
                )

            return PipelineOutput(
                ok=True,
                status="accepted",
                category=f"{top_name} ({top_code.upper()})",
                decision_code=top_code.upper(),
                route=r_res.route,
                route_name=r_res.route_name,
                target_pipeline=r_res.target_pipeline,
                confidence=round(top_score, 4),
                message=EXPLAIN.get(top_code, f"Predicted class: {top_name}"),
                warnings=pipeline_warnings,
                scores=scores,
                melanoma_flag=mel_flag,
                melanoma_flag_info=mel_info,
                quality_result=q_res,
                domain_result=d_res,
                routing_result=r_res,
                annotated_image=img_bgr,
            )

        # ================= ROUTE B & C: CLINICAL APPEARANCE (HAND / MACRO) =================
        hand_bbox = r_res.hand_crop_bbox
        is_hand = (r_res.route == "hand_photo")

        clin_res = self.clinical_classifier.classify(
            img_bgr=img_bgr,
            hand_bbox=hand_bbox,
            is_hand=is_hand,
        )

        all_warnings = pipeline_warnings + clin_res.warnings

        return PipelineOutput(
            ok=True,
            status=clin_res.status,
            category=clin_res.category_display,
            decision_code=clin_res.decision,
            route=r_res.route,
            route_name=r_res.route_name,
            target_pipeline=r_res.target_pipeline,
            confidence=clin_res.confidence,
            message=clin_res.message,
            warnings=all_warnings,
            clinical_result=clin_res,
            quality_result=q_res,
            domain_result=d_res,
            routing_result=r_res,
            annotated_image=r_res.annotated_image,
            cropped_image=clin_res.cropped_image,
        )
