"""Normal hand appearance modeling and anomaly detection package.

Educational research prototype - NOT a medical device.
Models typical human hand appearance and detects inputs that differ from the reference distribution.
"""

from .cropper import HandCropper, CropResult
from .feature_extractor import HandFeatureExtractor
from .anomaly_detector import MahalanobisAnomalyDetector, KNNAnomalyDetector
from .pipeline import NormalHandPipeline, HandInspectionResult

__all__ = [
    "HandCropper",
    "CropResult",
    "HandFeatureExtractor",
    "MahalanobisAnomalyDetector",
    "KNNAnomalyDetector",
    "NormalHandPipeline",
    "HandInspectionResult",
]
