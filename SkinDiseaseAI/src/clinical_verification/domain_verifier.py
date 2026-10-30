"""Stage 2 — Skin and Image Domain Verification.

Educational research prototype — NOT a medical diagnostic device.
Determines whether an input image contains supported human skin or should be rejected.
Screens against:
- Non-skin objects (cars, bikes, buildings, animals, household objects, electronics).
- Artificial images, drawings, cartoons, gaming posters, and anime.
- Presentation attacks (digital screens, printed replays, specular glare).
- Insufficient skin tissue coverage.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Semantic non-skin keyword filters for ImageNet-1k labels
NON_SKIN_OBJECT_KEYWORDS = {
    # Vehicles & Transportation
    "car", "automobile", "convertible", "sports_car", "racer", "pickup", "minivan", "cab", "limousine",
    "jeep", "landrover", "truck", "trailer", "tow_truck", "fire_engine", "garbage_truck",
    "motorcycle", "moped", "motor_scooter", "scooter", "bicycle", "bike", "tricycle",
    "train", "locomotive", "railcar", "airplane", "airliner", "wing", "space_shuttle",
    "boat", "ship", "speedboat", "canoe", "yacht", "lifeboat", "submarine",
    # Buildings & Architecture
    "church", "palace", "castle", "monastery", "barn", "boathouse", "beacon", "lighthouse",
    "dam", "dock", "pier", "bridge", "viaduct", "greenhouse", "cinema", "planetarium",
    # Animals & Wildlife
    "dog", "cat", "terrier", "hound", "retriever", "spaniel", "shepherd", "bird", "eagle",
    "hawk", "owl", "parrot", "fish", "shark", "whale", "snake", "lizard", "frog", "toad",
    "turtle", "tiger", "lion", "bear", "elephant", "zebra", "horse", "cow", "ox", "sheep",
    "monkey", "ape", "gorilla", "spider", "beetle", "butterfly",
    # Electronics, Appliances & Furniture
    "laptop", "notebook", "desktop_computer", "monitor", "screen", "television", "tv",
    "cellular_telephone", "phone", "dial_telephone", "keyboard", "mouse", "modem",
    "refrigerator", "microwave", "dishwasher", "toaster", "vacuum", "washing_machine",
    "chair", "desk", "table", "couch", "sofa", "bed", "wardrobe", "bookcase", "cabinet",
    # Everyday non-skin items
    "coffee_mug", "cup", "bottle", "wine_bottle", "beer_glass", "plate", "fork", "knife",
    "guitar", "violin", "piano", "drum", "traffic_light", "street_sign", "parking_meter",
    # Graphic, Print & 2D media
    "comic_book", "book_jacket", "envelope", "packet", "carton", "jigsaw_puzzle", "web_site",
    "menu", "crossword_puzzle", "banner", "poster",
}


@dataclass
class DomainVerificationResult:
    passed: bool
    status: str  # "accepted", "rejected", "uncertain"
    stage_failed: Optional[str] = None  # "non_skin_object", "drawing_or_anime", "skin_coverage", "spoof_check"
    domain_type: str = "Unknown"  # "genuine_skin_region", "non_skin_object", "drawing_or_anime", "insufficient_skin_coverage", "suspected_spoof"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skin_fraction: float = 0.0
    top_semantic_predictions: List[Dict[str, Any]] = field(default_factory=list)
    is_drawing_or_anime: bool = False
    drawing_score: float = 0.0
    spoof_analysis: Dict[str, Any] = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None


@lru_cache(maxsize=1)
def _get_imagenet_classifier():
    """Lazily load MobileNetV3Small with ImageNet top classification weights."""
    import tensorflow as tf
    model = tf.keras.applications.MobileNetV3Small(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3),
    )
    return model


class SkinDomainVerifier:
    """Stage 2 Multi-modal skin and domain verification engine."""

    def __init__(
        self,
        min_skin_fraction: float = 0.15,
        object_rejection_threshold: float = 0.35,
        drawing_rejection_threshold: float = 0.65,
    ):
        self.min_skin_fraction = min_skin_fraction
        self.object_rejection_threshold = object_rejection_threshold
        self.drawing_rejection_threshold = drawing_rejection_threshold

    # ---------- 1. Semantic Object Classification (ImageNet) ----------
    def evaluate_semantic_domain(self, pil_rgb: Image.Image) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """Classify input against ImageNet-1k to identify cars, bikes, buildings, everyday objects."""
        from tensorflow.keras.applications.mobilenet_v3 import decode_predictions

        model = _get_imagenet_classifier()
        im_resized = pil_rgb.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(im_resized, dtype=np.float32), 0)

        preds = model.predict(arr, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]

        top_preds = []
        for class_id, class_name, score in decoded:
            top_preds.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(score), 4),
            })

        # Check top-1 and top-3 for explicit non-skin objects
        top1 = top_preds[0]
        name1 = top1["class_name"].lower()
        conf1 = top1["confidence"]

        # Match against non-skin keywords
        is_top1_non_skin = any(kw in name1 for kw in NON_SKIN_OBJECT_KEYWORDS)

        # Calculate cumulative non-skin confidence in top-3
        non_skin_cum_conf = sum(
            p["confidence"] for p in top_preds[:3]
            if any(kw in p["class_name"].lower() for kw in NON_SKIN_OBJECT_KEYWORDS)
        )

        if (is_top1_non_skin and conf1 >= self.object_rejection_threshold) or (non_skin_cum_conf >= 0.55):
            human_readable = name1.replace("_", " ").title()
            reason = f"Non-skin object detected: '{human_readable}' (confidence: {conf1:.1%}). Please upload a clinical skin photograph."
            return False, reason, top_preds

        return True, None, top_preds

    # ---------- 2. Skin Chrominance & Texture Segmentation ----------
    def segment_human_skin(self, img_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute pixel mask and coverage fraction of physiological skin tissue."""
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return 0.0, np.zeros((1, 1), dtype=np.uint8)

        # 1. YCrCb chromaticity
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(combined_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        skin_fraction = float((clean_mask > 0).mean())

        return skin_fraction, clean_mask

    # ---------- 3. Drawing, Cartoon & Anime Detector ----------
    def detect_drawing_or_anime(self, img_bgr: np.ndarray, top_preds: List[Dict[str, Any]]) -> Tuple[bool, float, List[str]]:
        """Identify digital illustrations, cartoons, anime art, or flat drawings."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        reasons = []
        score = 0.0

        # Heuristic A: ImageNet predicted graphic / cartoon class
        for p in top_preds[:2]:
            cname = p["class_name"].lower()
            if any(term in cname for term in ["comic_book", "carton", "envelope", "packet", "jigsaw_puzzle", "web_site", "menu", "book_jacket"]):
                if p["confidence"] >= 0.25:
                    score += 0.50
                    reasons.append(f"Semantic structure matches artificial graphic or illustration ('{cname.replace('_', ' ')}')")
                    break

        # Heuristic B: Color quantization & low color variance in local regions (flat cartoon shading)
        # Resize to fixed size for fast color diversity analysis
        thumb = cv2.resize(img_bgr, (128, 128))
        unique_colors = len(np.unique(thumb.reshape(-1, 3), axis=0))
        # Natural skin photos have > 4,000 unique colors in 128x128 thumbnail; flat art often has < 1,500
        if unique_colors < 1800:
            score += 0.35
            reasons.append(f"Flat synthetic color palette detected ({unique_colors} unique colors vs >4000 in real skin)")

        # Heuristic C: Cartoon Line Art / Edge-to-Texture ratio
        # Anime/drawings have extremely sharp Canny edges separating completely flat regions
        edges = cv2.Canny(gray, 60, 180)
        edge_density = float((edges > 0).mean())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if edge_density > 0.08 and lap_var < 80.0:
            score += 0.25
            reasons.append("Sharp artificial line contours on flat fills detected (anime/drawing hallmark)")

        # Heuristic D: Unnatural saturation distribution (anime often uses pure neon/primary tones)
        sat = hsv[:, :, 1]
        extreme_sat_frac = float((sat > 200).mean())
        if extreme_sat_frac > 0.30:
            score += 0.25
            reasons.append("Extreme non-physiological color saturation detected")

        is_drawing = score >= self.drawing_rejection_threshold
        return is_drawing, round(score, 3), reasons

    # ---------- 4. Anti-Spoofing & Replay Check ----------
    def check_spoof_and_authenticity(
        self,
        img_bgr: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Screen for physical presentation attacks: screens, moiré grids, display bezels, flash glare."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if roi_bbox is not None:
            bx, by, bw, bh = roi_bbox
            bx1, by1 = max(0, bx), max(0, by)
            bx2, by2 = min(w, bx + bw), min(h, by + bh)
            roi_gray = gray[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else gray
        else:
            roi_gray = gray

        # 2D FFT Spectral Analysis
        f = np.fft.fft2(cv2.resize(roi_gray, (256, 256)))
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-7)

        ch, cw = 128, 128
        y_idx, x_idx = np.ogrid[:256, :256]
        dist = np.sqrt((x_idx - cw)**2 + (y_idx - ch)**2)
        high_freq_mask = (dist > 60) & (dist < 120)
        high_freq_energy = float(magnitude[high_freq_mask].mean())
        center_energy = float(magnitude[dist <= 30].mean())
        spectral_ratio = high_freq_energy / (center_energy + 1e-5)

        # Rectilinear border detection (phone / laptop screen bezel)
        edges = cv2.Canny(roi_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=int(min(roi_gray.shape[:2]) * 0.4), maxLineGap=10)
        num_border_lines = 0
        has_horiz = False
        has_vert = False
        rh, rw = roi_gray.shape[:2]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.ravel()[:4]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                if dy < 4 and dx > (0.40 * rw):
                    # Check if line is close to border (screen bezel)
                    if y1 < 0.20 * rh or y1 > 0.80 * rh:
                        num_border_lines += 1
                        has_horiz = True
                elif dx < 4 and dy > (0.40 * rh):
                    if x1 < 0.20 * rw or x1 > 0.80 * rw:
                        num_border_lines += 1
                        has_vert = True

        has_screen_frame = has_horiz and has_vert and (num_border_lines >= 2)

        # Specular reflection fraction
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        extreme_specular_frac = float(((sat < 20) & (val > 245)).mean())

        suspicions = []
        if spectral_ratio > 0.85:
            suspicions.append("Periodic high-frequency pattern detected (screen moiré grid or print halftoning)")
        if has_screen_frame:
            suspicions.append("Rectilinear display bezel or photo frame detected")
        if extreme_specular_frac > 0.20:
            suspicions.append("Severe screen glass or display reflection detected")

        authenticity = "suspected_spoof" if suspicions else "unverified_plausible"
        return {
            "authenticity": authenticity,
            "suspicions": suspicions,
            "spectral_ratio": round(spectral_ratio, 3),
            "border_line_count": num_border_lines,
            "specular_fraction": round(extreme_specular_frac, 3),
        }

    # ---------- Master Stage 2 Verification ----------
    def verify(
        self,
        img_bgr: np.ndarray,
        pil_rgb: Image.Image,
        is_hand_detected: bool = False,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> DomainVerificationResult:
        """Run complete Stage 2 domain verification."""
        # 1. Semantic Object Classification
        sem_ok, sem_reason, top_preds = self.evaluate_semantic_domain(pil_rgb)
        if not sem_ok:
            return DomainVerificationResult(
                passed=False,
                status="rejected",
                stage_failed="non_skin_object",
                domain_type="non_skin_object",
                reasons=[sem_reason],
                top_semantic_predictions=top_preds,
            )

        # 2. Skin Chrominance & Coverage
        skin_frac, skin_mask = self.segment_human_skin(img_bgr)

        # Contour overlay visualization
        annotated = img_bgr.copy()
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 220, 0), 2)

        # 3. Drawing / Anime / Cartoon check
        is_drawing, draw_score, draw_reasons = self.detect_drawing_or_anime(img_bgr, top_preds)
        if is_drawing:
            return DomainVerificationResult(
                passed=False,
                status="rejected",
                stage_failed="drawing_or_anime",
                domain_type="drawing_or_anime",
                reasons=["Artificial drawing or anime character detected: " + "; ".join(draw_reasons)],
                skin_fraction=round(skin_frac, 3),
                top_semantic_predictions=top_preds,
                is_drawing_or_anime=True,
                drawing_score=draw_score,
                annotated_image=annotated,
            )

        # 4. Anti-Spoofing Check
        spoof_res = self.check_spoof_and_authenticity(img_bgr, roi_bbox=roi_bbox)
        if spoof_res["authenticity"] == "suspected_spoof":
            return DomainVerificationResult(
                passed=False,
                status="rejected",
                stage_failed="spoof_check",
                domain_type="suspected_spoof",
                reasons=["Suspicious presentation attack / spoof detected: " + "; ".join(spoof_res["suspicions"])],
                skin_fraction=round(skin_frac, 3),
                top_semantic_predictions=top_preds,
                spoof_analysis=spoof_res,
                annotated_image=annotated,
            )

        # 5. Check Skin Coverage Threshold (unless an isolated hand landmark was confirmed)
        if skin_frac < self.min_skin_fraction and not is_hand_detected:
            return DomainVerificationResult(
                passed=False,
                status="rejected",
                stage_failed="skin_coverage",
                domain_type="insufficient_skin_coverage",
                reasons=[f"No skin tissue detected (skin coverage: {skin_frac:.1%}, minimum {self.min_skin_fraction:.0%} required). Please upload a clear skin photograph."],
                skin_fraction=round(skin_frac, 3),
                top_semantic_predictions=top_preds,
                spoof_analysis=spoof_res,
                annotated_image=annotated,
            )

        warnings = []
        if draw_score >= 0.40:
            warnings.append(f"Moderate graphic-art pattern score ({draw_score:.2f}). Ensure the photo is of genuine human skin.")
        if spoof_res["authenticity"] == "unverified_plausible":
            warnings.append("Authenticity Notice: Anti-spoofing heuristics found no obvious screen/print artifacts, but physical presentation cannot be guaranteed without dedicated hardware.")

        return DomainVerificationResult(
            passed=True,
            status="accepted",
            stage_failed=None,
            domain_type="genuine_skin_region",
            reasons=[],
            warnings=warnings,
            skin_fraction=round(skin_frac, 3),
            top_semantic_predictions=top_preds,
            is_drawing_or_anime=False,
            drawing_score=draw_score,
            spoof_analysis=spoof_res,
            annotated_image=annotated,
        )
