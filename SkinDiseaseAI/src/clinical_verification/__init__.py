"""Clinical Skin Verification & Classification Pipeline Package.

Educational research prototype - NOT a medical diagnostic device.
Provides modular stages for image quality verification, skin/domain verification,
anatomical routing, clinical appearance classification, and uncertainty calibration.
"""

from .quality import ImageQualityChecker, QualityCheckResult
from .domain_verifier import SkinDomainVerifier, DomainVerificationResult
from .router import AnatomicalRouter, RoutingResult
from .clinical_classifier import ClinicalSkinClassifier, ClinicalClassificationResult
from .pipeline import ClinicalVerificationPipeline, PipelineOutput

__all__ = [
    "ImageQualityChecker",
    "QualityCheckResult",
    "SkinDomainVerifier",
    "DomainVerificationResult",
    "AnatomicalRouter",
    "RoutingResult",
    "ClinicalSkinClassifier",
    "ClinicalClassificationResult",
    "ClinicalVerificationPipeline",
    "PipelineOutput",
]
