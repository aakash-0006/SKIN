# Integration Plan: Verified Hand Anomaly Baseline into SkinDiseaseAI

**Date:** October 29, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Target System:** SkinDiseaseAI Streamlit Application & Inference Pipeline  
**Model Artifact:** [`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/hand_anomaly_baseline.npz)  
**Baseline Technical Report:** [`reports/experiments/normal_hand_v1/stage_d_baseline_report.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/normal_hand_v1/stage_d_baseline_report.md)  
**Current Milestone:** Stage E — Integration Architecture and Implementation Planning  

---

## 1. Pre-Integration Verification of Repository Components

Prior to architectural design, all physical paths, module interfaces, and inference dependencies were verified against the active codebase:

### A. Existing Seven-Class Lesion Classifier
* **Model Checkpoint:** `models/best_model.keras` (SHA-256: `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd`, 100% verified).
* **Augmented Follow-up Checkpoint:** `models/isic_augmented_v1.keras` (SHA-256: `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34`, 100% verified).
* **Supported Diagnostic Classes:** `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
* **Input Modality:** Strictly **close-up dermoscopy** ($224 \times 224 \times 3$, normalized $[0, 1]$).
* **Current Interception Behavior:** In [src/predict.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py) (lines 154–182), when `val_res.hand_count >= 1`, the disease classifier is already strictly intercepted and suppressed (`code = None, scores = []`).

### B. Verified Normal-Hand Anomaly Detector
* **Model Artifact:** `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` (SHA-256: `5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4`).
* **Localization & Cropping:** [src/normal_hand/cropper.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/cropper.py) (`HandCropper`). Uses MediaPipe Hands with a $10\%$ safety padding margin and boundary clamping.
* **Feature Extraction:** [src/normal_hand/feature_extractor.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/feature_extractor.py) (`HandFeatureExtractor`). Frozen ImageNet-pretrained MobileNetV3Small backbone, global average pooling, output dimension $576$.
* **Scoring Engine:** [src/normal_hand/anomaly_detector.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/anomaly_detector.py) (`MahalanobisAnomalyDetector`). PCA ($k=48$ components, $83.87\%$ variance explained) with shrinkage regularized covariance ($\lambda = 0.05$).
* **Calibrated Decision Threshold:** Locked strictly on validation normal data: **`t = 8.768`**.
* **Pipeline API:** [src/normal_hand/pipeline.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/pipeline.py) (`NormalHandPipeline`).

---

## 2. Combined Workflow Architecture & Routing Logic

The integrated pipeline prevents domain confounding by enforcing strict architectural separation between **dermoscopic lesion analysis** and **macro-photo hand appearance evaluation**.

### A. Modular Routing Logic Diagram

```
                                [ Uploaded Image Bytes ]
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │ STAGE 1: File Integrity & Input Validation       │
                 │ • Check format (JPEG/PNG), resolution, blur, lum │
                 └─────────────────────────┬────────────────────────┘
                                           │ Passed Quality
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │ STAGE 2: Modality & Anatomical Routing Gate      │
                 │ • MediaPipe Landmark Localization                │
                 │ • Skin Chromaticity Tissue Verification          │
                 │ • Presentation Attack / Anti-Spoof Screening     │
                 └─────────────────────────┬────────────────────────┘
                                           │
                 ┌─────────────────────────┴────────────────────────┐
                 ▼                                                  ▼
     [ Hand Detected (>= 1 Hand) ]                    [ No Hand Detected / Close-up ]
                 │                                                  │
                 ▼                                                  ▼
┌─────────────────────────────────┐                ┌─────────────────────────────────┐
│ STAGE 3A: Normal Hand Pipeline  │                │ STAGE 3B: Dermoscopy Domain Gate│
│ • HandCropper (10% padding)     │                │ • Check Mahalanobis dermoscopy  │
│ • MobileNetV3Small (576-dim)    │                │   distance in ood_gate.npz      │
│ • PCA-Mahalanobis (k=48)        │                └────────────────┬────────────────┘
│ • Threshold: t = 8.768          │                                 │
└────────────────┬────────────────┘                                 ▼
                 │                                 ┌─────────────────────────────────┐
                 │                                 │ STAGE 4: Seven-Class Classifier │
                 │                                 │ • EfficientNetB0 / Ensemble     │
                 │                                 │ • Output: akiec..vasc + alert   │
                 │                                 └────────────────┬────────────────┘
                 │                                                  │
                 ▼                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Result-Composition & Medical Guardrail Layer                              │
│ • Hand Branch: Emits USABLE_HAND_IMAGE or UNUSUAL_RELATIVE_TO_TRAINING_DATA        │
│   (Lesion classifier is explicitly SUPPRESSED as domain-incompatible)              │
│ • Dermoscopy Branch: Emits disease probability distribution + abstain rule         │
│   (Hand appearance model is marked NOT APPLICABLE)                                 │
│ • Clear Medical Disclaimers: Neither branch certifies a user is disease-free       │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### B. Routing Invariants
1. **Invariant 1 (Strict Lesion Suppression on Macro Hands):** Under no circumstance is a macro camera photo of a hand passed to the 7-class lesion classifier. The 7-class model was trained on dermoscopic close-ups; running it on non-dermoscopic photos produces meaningless, uncalibrated classifications.
2. **Invariant 2 (Strict Hand Anomaly Suppression on Dermoscopy):** Dermoscopic close-ups of lesions on other body parts (torso, back, legs) contain zero hand landmarks and must bypass the hand anomaly pipeline entirely.
3. **Invariant 3 (No False Diagnostic Claims):** An anomaly score below $8.768$ never produces a `HEALTHY` or `NO DISEASE` tag. An anomaly score above $8.768$ never produces a `DISEASE DETECTED` tag.

---

## 3. Comprehensive Limitation & Risk Assessment

Integration must explicitly communicate the empirical findings established during Stage D evaluation. These numbers are non-negotiable and must not be obscured:

### Measured Subgroup Disparities (Held-Out Test Cohort, $N=420$)
* **Overall Baseline False Rejection Rate:** **`10.24%`** ($43 / 420$ typical normal hands flagged as unusual).
* **The Accessories / Jewellery Effect:**
  * Bare hands without accessories: **`4.35%`** false rejection ($11 / 253$).
  * Hands wearing rings/watches: **`19.16%`** false rejection ($32 / 167$).
  * **Risk:** Metallic reflections, gemstone highlights, and dark watchbands shift feature embeddings out of the normal skin density manifold, **quadrupling** the false positive rate.
* **The Fingernail Polish Effect:**
  * Unpainted fingernails: **`8.95%`** false rejection ($34 / 380$).
  * Painted fingernails: **`22.50%`** false rejection ($9 / 40$).
  * **Risk:** Artificial enamel pigments on nail beds distort color histograms and trigger false anomaly alerts.
* **Demographic Age Skew:**
  * Median training age was **21.0 years**. Geriatric hands naturally exhibiting loss of subcutaneous fat, prominent superficial dorsal veins, and solar lentigines will experience elevated anomaly scores.

### UX Mitigation Protocol (Without False Claims of Resolution)
1. **Pre-Capture Guidance Card:**  
   Before image upload, the UI will display a prominent advisory:  
   *"For optimal accuracy: Remove wristwatches, bracelets, and rings. Ensure even diffuse lighting without harsh shadows."*
2. **Context-Aware Dynamic Warnings:**  
   If an uploaded hand is flagged as `UNUSUAL_RELATIVE_TO_TRAINING_DATA` and accessories or polished nails are visually detected, the UI will display:  
   *“Notice: Visible jewellery or nail polish was detected. In our benchmark, accessories quadrupled false anomaly alerts (19.2% vs 4.4%). Please remove jewellery and re-test before interpreting this result.”*
3. **Explicit Handling of Cropping Failures:**  
   If hand localization fails, the UI will output `UNSUITABLE_OR_UNCERTAIN_INPUT` and show the failure reason (e.g. `RESOLUTION_TOO_LOW`, `NO_HAND_DETECTED`) with a troubleshooting guide.

---

## 4. Proposed Streamlit UI Architecture

The Streamlit interface in [app.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py) will be upgraded with a modular four-section results layout:

```
========================================================================================
🩺 SkinDiseaseAI — Input Verification & Analysis
========================================================================================

[ Step 1: Input Quality ] ──── [ Step 2: Modality Gate ] ──── [ Step 3: Analysis ]
      ✅ Usable                    ✋ Macro Hand Photo            ⚠️ Review Complete

────────────────────────────────────────────────────────────────────────────────────────
SECTION 1: INPUT MODALITY & QUALITY VERIFICATION
────────────────────────────────────────────────────────────────────────────────────────
• Detected Modality: Hand Macro-Photograph (Non-Dermoscopic Camera Photo)
• Quality Metrics: Resolution: 1600x1200 | Focus (Laplacian): 184.2 | Luminance: 142.1
• Cropping Status: Hand localized successfully (Crop size: 840x1120)

────────────────────────────────────────────────────────────────────────────────────────
SECTION 2: NORMAL-HAND APPEARANCE ANALYSIS (MobileNetV3 PCA-Mahalanobis)
────────────────────────────────────────────────────────────────────────────────────────
Status: [ ⚠️ UNUSUAL RELATIVE TO TRAINING DATA ]  or  [ ✅ USABLE HAND IMAGE ]

• Anomaly Distance: 9.42  (Reference Normal Threshold: 8.77)
• Interpretation: The visual appearance of this hand differs from the reference normal
  training dataset. This variation may be caused by a dermatological condition, unusual
  skin texture, strong shadowing, wristwatches, or rings.
• Subgroup Advisory: Wearing jewellery or nail polish elevates false anomaly rates
  (up to 19.2% - 22.5%). If jewellery is present, remove it and re-scan.

[ Cropped Hand Overlay ]           [ Technical Distribution Expander ]
┌─────────────────────────┐        • Metric: PCA-Mahalanobis (k=48, shrinkage=0.05)
│  (Hand region only,     │        • Backbone: Frozen MobileNetV3Small (576-dim)
│   white borders         │        • Reference Cohort: 11k Hands (129 participants)
│   excluded)             │        • Distance Margin: +0.65 above threshold
└─────────────────────────┘

────────────────────────────────────────────────────────────────────────────────────────
SECTION 3: SEVEN-CLASS LESION CLASSIFIER (EfficientNetB0 Ensemble)
────────────────────────────────────────────────────────────────────────────────────────
Status: 🛑 NOT APPLICABLE FOR THIS IMAGE TYPE

• Notice: The 7-class lesion classifier (melanoma, basal cell carcinoma, nevi, etc.)
  requires high-magnification dermoscopic close-up imagery.
• It is intentionally SUPPRESSED for macro-camera hand photos to prevent unvalidated,
  misleading diagnostic predictions.

────────────────────────────────────────────────────────────────────────────────────────
SECTION 4: MANDATORY CLINICAL LIMITATIONS & SAFETY DISCLAIMER
────────────────────────────────────────────────────────────────────────────────────────
⚠️ EDUCATIONAL PROTOTYPE — NOT A MEDICAL DEVICE:
1. Appearance consistent with normal hand data DOES NOT certify that the skin is healthy
   or free from dermatological disease.
2. An unusual appearance score DOES NOT constitute a medical diagnosis of any condition.
3. Consult a qualified dermatologist or medical practitioner for any skin symptom.
========================================================================================
```

---

## 5. Scope of Implementation: Exact Files Requiring Modification

To implement this plan safely, modifications will be restricted to **two files** in the application layer:

| File Path | Nature of Modification | Description of Change |
| :--- | :--- | :--- |
| [src/predict.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py) | Internal Pipeline Integration | 1. Import `NormalHandPipeline` from `src.normal_hand.pipeline`.<br>2. When `val_res.hand_count >= 1`, invoke `NormalHandPipeline.inspect_image()`.<br>3. Populate `hand_analysis` dictionary in return payload with anomaly distance, threshold, decision, and cropped overlay.<br>4. Keep 7-class disease classifier suppressed on hand photos. |
| [app.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py) | Streamlit User Interface | 1. Update pipeline stepper column 5 to show normal-hand appearance status.<br>2. Add dedicated Section 2 (Hand Appearance Analysis with metrics, threshold card, and advisory).<br>3. Add explicit suppression card for Section 3 (Lesion Classifier).<br>4. Render unified limitations banner. |

### Untouched Protected Assets
* `models/best_model.keras` — UNTOUCHED
* `models/isic_finetuned.keras` — UNTOUCHED
* `models/isic_augmented_v1.keras` — UNTOUCHED
* `models/ood_gate.npz` — UNTOUCHED
* `outputs/splits.csv` & `outputs/combined_splits.csv` — UNTOUCHED
* `config.json` — UNTOUCHED
* Source datasets & `HandInfo.csv` — UNTOUCHED

---

## 6. Required Verification Tests Before Production Sign-Off

The integration pull request must satisfy the following automated verification suite:

1. **Test Case 1 (Typical Hand Photo):** Input an authentic hand photo without jewellery.
   * *Assert:* Hand detected; Hand anomaly pipeline invoked; Distance scored; Decision is `USABLE_HAND_IMAGE` or `UNUSUAL_RELATIVE_TO_TRAINING_DATA`; 7-class classifier is suppressed (`code is None, scores == []`).
2. **Test Case 2 (Dermoscopic Lesion Image):** Input an authentic HAM10000 dermoscopy image.
   * *Assert:* Domain gate passes; Hand detector reports 0 hands; Hand anomaly pipeline bypassed; 7-class classifier is executed and produces lesion probabilities.
3. **Test Case 3 (Classroom / Out-of-Domain Webcam Photo):** Input `tests/fixtures/webcam_classroom_hand.jpg`.
   * *Assert:* Domain gate rejects input ($d > 2,673.1$); Pipeline halts with `ok = False`; Zero disease scores emitted.
4. **Test Case 4 (Vocabulary Guardrail):** Execute 100 test queries through `predict_bytes`.
   * *Assert:* Zero responses contain the strings `"HEALTHY"`, `"DISEASE DETECTED"`, or `"NO DISEASE"`.
5. **Test Case 5 (Latency Budget):** Measure end-to-end processing time for a $1600 \times 1200$ hand photo.
   * *Assert:* Total wall-clock time $\le 200\text{ ms}$ on CPU.

---

## 7. Rollback Strategy

Because all model artifacts and preprocessing logic are fully encapsulated within `src/normal_hand/`:
1. **Zero Asset Coupling:** The existing lesion classifier does not depend on `src/normal_hand/`.
2. **Feature Toggle / Instant Rollback:** In [src/predict.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py), the hand anomaly invocation will be wrapped in a feature flag:
   ```python
   ENABLE_NORMAL_HAND_DETECTOR = True
   ```
   If set to `False`, `predict_bytes()` immediately reverts to its previous behavior (returning the static message: *"Hand detected. This system cannot yet tell normal skin from a skin condition."*).
3. **Clean Git Reversion:** The proposed modifications touch only `src/predict.py` and `app.py`. A single `git checkout` on these two files restores the system to its exact pre-integration state within 5 seconds.

---

## 8. Checkpoint Status

**Stage E planning is complete.**
* The complete integration plan is documented and sealed.
* **No code modifications or live app changes have been made.**
* **Action:** STOPPING and awaiting user approval before beginning implementation.
