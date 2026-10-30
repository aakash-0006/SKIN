# Stage F: Controlled Integration Implementation and Verification Report

**Date:** October 29, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Target System:** SkinDiseaseAI Streamlit Application & Inference Pipeline  
**Model Checkpoints:**
- Primary 7-Class Dermoscopic Classifier: `models/best_model.keras` (SHA-256: `c6af0869...`, protected & untouched)
- Normal-Hand Appearance Anomaly Detector: `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` (SHA-256: `5b2b5345...`, protected & untouched)
- Dermoscopy OOD Gate: `models/ood_gate.npz` (SHA-256: `dcd9974a...`, protected & untouched)

**Status:** Implementation & Verification Complete. Feature flag is **DISABLED (`ENABLE_NORMAL_HAND_DETECTOR = False`)**. System is stopped and awaiting explicit user approval before live activation.

---

## 1. Executive Summary & Verification of Constraints

In accordance with Stage F instructions and the approved Stage E Integration Plan (`reports/experiments/normal_hand_v1/integration_plan.md`), the Stage D verified baseline was integrated into the application layer under strict isolation constraints:

1. **Pre-Change Rollback Point Created:**
   - Dedicated backup directory: `backups/stage_f_integration_backup/`
   - Pre-edit backups saved: `predict.py.bak` and `app.py.bak`.
   - Hashes of all 7 protected assets recorded before any edits and confirmed 100% bitwise identical after all edits.
2. **File Boundary Enforcement:**
   - **Only two files modified:** `src/predict.py` and `app.py`.
   - Zero modifications to core model files, training pipelines, manifest files, or the quarantined exploratory cohort ($N=243$).
3. **Exact Stage D Methodology Reused:**
   - MobileNetV3Small frozen backbone ($576$-dimensional embedding).
   - $k=48$ PCA components ($83.87\%$ variance explained), shrinkage regularized covariance ($\lambda = 0.05$).
   - Exact validation-calibrated decision threshold locked at **`t = 8.768`** (never retuned or modified).
4. **Modality-Specific Exclusive Routing:**
   - Macro-camera hand photographs are routed **exclusively** through the normal-hand appearance pipeline.
   - Dermoscopic images are routed **exclusively** through the existing domain gate and seven-class classifier.
   - The seven-class classifier is strictly suppressed for macro hand photos (`code = None`, `scores = []`, `melanoma_flag = False`).
5. **Feature Flag Defaults to Disabled:**
   - `ENABLE_NORMAL_HAND_DETECTOR = False` in `src/predict.py`.
   - When disabled, application behavior reverts exactly to previous baseline behavior.
   - All 47 automated tests in the test suite pass with 0 failures and 0 errors.

---

## 2. Protected Asset SHA-256 Hash Comparisons

All protected repository assets were verified against the locked pre-experiment backup (`D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip`). Hashes remain 100% bitwise identical:

| Asset Path | Expected Hash (Locked Backup) | Post-Integration Hash | Status |
| :--- | :--- | :--- | :--- |
| `models/best_model.keras` | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **MATCH (100% Identical)** |
| `models/isic_finetuned.keras` | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **MATCH (100% Identical)** |
| `models/isic_augmented_v1.keras` | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **MATCH (100% Identical)** |
| `models/ood_gate.npz` | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **MATCH (100% Identical)** |
| `outputs/splits.csv` | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **MATCH (100% Identical)** |
| `outputs/combined_splits.csv` | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **MATCH (100% Identical)** |
| `config.json` | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **MATCH (100% Identical)** |
| `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` | `5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4` | `5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4` | **MATCH (100% Identical)** |

---

## 3. Code Modifications Detailed Walkthrough

Only two files were modified, both strictly within the application interface layer:

### A. [`src/predict.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py)
1. **Feature Toggle Variable:**
   ```python
   # Feature toggle for normal hand anomaly detector: defaults to False (disabled) as required
   ENABLE_NORMAL_HAND_DETECTOR = False
   ```
2. **Lazy Pipeline Loader (`get_normal_hand_pipeline`):**
   - Implements `@lru_cache(maxsize=1)` lazy loading to avoid importing TensorFlow or loading baseline weights on initial module startup.
   - Validates that `hand_anomaly_baseline.npz` exists, is fitted, and has the locked Stage D threshold `8.768`.
3. **Exclusive Macro Hand Routing:**
   - When `ENABLE_NORMAL_HAND_DETECTOR` is enabled:
     - Detects if an image contains a hand via `pipeline.cropper.crop_hand(img_bgr)`.
     - If hand detected (`crop_res.success` or `hand_count >= 1`), executes `NormalHandPipeline.inspect_image()`.
     - Populates the `hand_analysis` payload with `decision`, `is_anomaly`, `anomaly_score`, `threshold = 8.768`, `crop_shape`, and cautious human-readable message.
     - Strictly intercepts and suppresses the 7-class lesion classifier (`code = None`, `scores = []`, `melanoma_flag = False`).
     - Bypasses the dermoscopic domain gate (`check_image`) to prevent false rejections of macro photos.
   - If no hand is detected (`hand_count == 0`), routes through the dermoscopic domain gate and the 7-class classifier (`best_model.keras`).
   - If `ENABLE_NORMAL_HAND_DETECTOR = False`, falls back immediately to the exact legacy execution path.

### B. [`app.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py)
1. **Four-Section Modular Hand Results Layout:**
   - When `hand_analysis` is present in the inference result, renders four clear, distinct sections:
     - **Section 1: Input Modality & Quality Verification:** Displays detected modality ("Hand photo (macro camera)"), hand count, localized crop dimensions, and quality metrics (focus variance, luminance, contrast).
     - **Section 2: Normal Hand Appearance Evaluation:** Displays the Mahalanobis anomaly distance metric card, calibrated threshold ($8.768$), and cautious decision banner:
       - `USABLE_HAND_IMAGE` $\rightarrow$ *"Consistent with Reference Normal Hand Skin Data: The visual appearance of this hand falls within the typical distribution of our reference cohort."*
       - `UNUSUAL_RELATIVE_TO_TRAINING_DATA` $\rightarrow$ *"Unusual Relative to Reference Normal Hand Data: The visual appearance of this hand differs from the reference normal hand training dataset."*
       - `UNSUITABLE_OR_UNCERTAIN_INPUT` $\rightarrow$ *"Unsuitable or Uncertain Input: Could not reliably localize or score a human hand in this image."*
     - **Subgroup Limitation Advisory:** Displays empirical confounding evidence:
       - Accessories/jewellery elevated false anomaly rate from $4.35\%$ to $19.16\%$ in testing.
       - Fingernail polish elevated false anomaly rate from $8.95\%$ to $22.50\%$.
       - Explicitly states: *"While removing accessories and photographing hands under even diffuse lighting reduces false flags, doing so does NOT guarantee a correct result."*
     - **Section 3: Seven-Class Lesion Classifier Status:** Prominent suppression notice explaining that the lesion model requires dermoscopy and is intentionally withheld for macro photos.
     - **Section 4: Mandatory Clinical Safety Notice:** Clearly states this is an educational research prototype and NOT a medical device. Consistent appearance does NOT certify skin is healthy; unusual score does NOT constitute a diagnosis.
2. **Legacy Fallback:**
   - If `ENABLE_NORMAL_HAND_DETECTOR` is False, renders the previous static warning message: *"Hand detected. This system cannot yet tell normal skin from a skin condition."*

---

## 4. Test Execution and Actual Results

### A. Stage F Regression & Integration Suite (`tests/test_stage_f_integration.py`)
Run command:
```powershell
& ".\.venv\Scripts\python.exe" -m unittest tests/test_stage_f_integration.py
```
Actual output:
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
.......
----------------------------------------------------------------------
Ran 8 tests in 5.078s

OK
```

**Individual Test Verification Breakdown:**
1. `test_1_feature_flag_defaults_to_false`: PASS. Verifies `predict_module.ENABLE_NORMAL_HAND_DETECTOR is False`.
2. `test_2_existing_seven_class_prediction_unaffected`: PASS. Verifies that 7-class lesion predictions on HAM10000 dermoscopy images are bitwise identical between flag `False` and flag `True` (`code="nv"`, `score=99.8%`, `melanoma_flag=False`).
3. `test_3_hand_images_never_receive_seven_class_prediction`: PASS. Verifies hand photographs receive `code = None`, `scores = []`, `melanoma_flag = False` in both flag states.
4. `test_4_hand_pipeline_uses_exact_threshold_and_preprocessing`: PASS. Verifies threshold is exactly $8.768$, feature dimension is $576$, PCA components $k=48$, and returned messages contain neither `"DISEASE"` nor `"HEALTHY"`.
5. `test_5_unsuitable_and_invalid_inputs_handled_safely`: PASS. Verifies empty bytes, corrupted garbage bytes, and unrelated gaming posters are rejected gracefully with `ok = False` without uncaught exceptions.
6. `test_6_protected_assets_hashes_remain_identical`: PASS. Verifies all 7 protected files match their locked hashes.
7. `test_7_missing_artifact_handled_gracefully`: PASS. Mocks missing model artifact and verifies pipeline returns clean error (`ok = False`, `stage_failed = "hand_model_missing"`) instead of crashing or returning misleading predictions.
8. `test_8_streamlit_interface_layout_and_disclaimer_present`: PASS. Verifies that all 4 required UI sections, measured limitation statistics ($19.16\%$ and $22.50\%$), and non-guarantee disclaimer exist in `app.py`.

### B. Full Repository Test Suite Discovery
Run command:
```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests
```
Actual output:
```
----------------------------------------------------------------------
Ran 47 tests in 11.018s

OK
```
- `tests/test_gate_quality.py`: 7 tests passed (image quality checks, corrupt handling, darkness, glare).
- `tests/test_normal_hand_baseline.py`: 11 tests passed (Stage D pipeline, cropper, feature extractor, PCA-Mahalanobis).
- `tests/test_regression_fixtures.py`: 2 tests passed (webcam classroom photo and gaming poster safely blocked).
- `tests/test_hand_validation_and_spoof.py`: 12 tests passed (skin detection, spoof screen/moiré detection, landmark validation).
- `tests/test_augmentation_and_weights.py`: 7 tests passed (capped class weights and evaluation data isolation).
- `tests/test_stage_f_integration.py`: 8 tests passed.

---

## 5. UI Layout and Presentation Specification

The integrated Streamlit user interface is structured to ensure medical safety, transparency, and clarity:

```
========================================================================================
🩺 SkinDiseaseAI — Input Verification & Analysis
========================================================================================

[ Step 1: Input Quality ] ──── [ Step 2: Modality Gate ] ──── [ Step 3: Analysis ]
      ✅ Usable                    ✋ Macro Hand Photo            ⚠️ Review Complete

────────────────────────────────────────────────────────────────────────────────────────
SECTION 1: INPUT MODALITY & QUALITY VERIFICATION
────────────────────────────────────────────────────────────────────────────────────────
• Detected Modality: Hand photo (macro camera)
• Hand Count: 1
• Localized Hand Crop Dimensions: 921 × 1120 px
• Focus Variance: 184.2 | Mean Luminance: 142.1 | Contrast: 64.3

────────────────────────────────────────────────────────────────────────────────────────
SECTION 2: NORMAL HAND APPEARANCE EVALUATION (MobileNetV3 PCA-Mahalanobis)
────────────────────────────────────────────────────────────────────────────────────────
┌─────────────────────────┬────────────────────────────────────────────────────────────┐
│ Hand Anomaly Distance   │  Status: ✅ Consistent with Reference Normal Hand Skin     │
│ 5.41                    │  Data                                                      │
│ -3.36 vs thr            │  The visual appearance of this hand falls within the       │
│ Calibrated Threshold:   │  typical distribution of our reference cohort.             │
│ 8.768 (Validation α=0.05)│                                                           │
└─────────────────────────┴────────────────────────────────────────────────────────────┘

⚠️ Confounder & Subgroup Advisory:
• Accessories / Jewellery: Wearing rings, wristwatches, or bracelets roughly quadrupled
  false anomaly alerts (19.16% with accessories vs. 4.35% without).
• Fingernail Polish: Painted fingernails elevated false anomaly alerts (22.50% vs 8.95%).
• Notice: While removing accessories and photographing hands under even diffuse lighting
  reduces false flags, doing so does NOT guarantee a correct result.

[ Technical Model Details Expander ]
• Metric: PCA-Regularized Mahalanobis Distance (k=48, shrinkage=0.05)
• Backbone: Frozen MobileNetV3Small (576-dim embedding, ImageNet weights)
• Reference Dataset: 11k Hands (129 participants)

────────────────────────────────────────────────────────────────────────────────────────
SECTION 3: SEVEN-CLASS LESION CLASSIFIER STATUS
────────────────────────────────────────────────────────────────────────────────────────
🛑 LESION CLASSIFIER NOT APPLICABLE FOR THIS IMAGE TYPE
The 7-class lesion classifier (melanoma, basal cell carcinoma, nevi, etc.) was trained
and validated exclusively on high-magnification dermoscopic close-ups of pigmented lesions.
To prevent misleading, unvalidated predictions, the classifier is intentionally
suppressed for non-dermoscopic camera photos of hands. No condition label, probability
scores, or melanoma alerts are produced.

────────────────────────────────────────────────────────────────────────────────────────
SECTION 4: MANDATORY CLINICAL SAFETY NOTICE
────────────────────────────────────────────────────────────────────────────────────────
🩺 EDUCATIONAL RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE:
1. Appearance consistent with normal hand data DOES NOT certify that the skin is healthy,
   disease-free, or without lesion.
2. An unusual appearance score DOES NOT constitute a medical diagnosis of any condition.
3. Consult a qualified dermatologist or clinician for any skin concern.
========================================================================================
```

---

## 6. Known Empirical Limitations & Clinical Guardrails

1. **Subgroup False Rejection Disparities:**
   - **Baseline False Rejection Rate:** $10.24\%$ ($43 / 420$) on typical normal hands.
   - **Accessories / Jewellery:** False rejection rate is **$19.16\%$** ($32/167$) with accessories vs. **$4.35\%$** ($11/253$) without.
   - **Fingernail Polish:** False rejection rate is **$22.50\%$** ($9/40$) with polish vs. **$8.95\%$** ($34/380$) without.
   - *Communication Rule:* The UI explicitly states that removing jewellery/polish reduces confounding but **does not guarantee a valid result**.
2. **Demographic & Environmental Skew:**
   - **Age Skew:** Training median age is **21.0 years**. Geriatric skin with loss of turgor, prominent veins, or actinic lentigines may naturally trigger elevated anomaly distances.
   - **Lighting & Uniform Background:** The baseline was fitted on studio photos against white backgrounds; complex natural backgrounds may produce cropping uncertainties.
3. **Quarantine Cohort Isolation:**
   - All 243 quarantined images across participants `27`, `570`, `1566`, and `1000000` remain 100% excluded from model fitting, validation, and testing.

---

## 7. Rollback Instructions

If rollback is ever required:
1. **Instant Rollback via Feature Toggle (< 1 second):**
   In [`src/predict.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py), line 41:
   ```python
   ENABLE_NORMAL_HAND_DETECTOR = False
   ```
   This immediately disables all hand anomaly detection logic and restores the exact pre-integration behavior.
2. **Full File Reversion (< 5 seconds):**
   Copy the versioned integration backups over the modified files:
   ```powershell
   Copy-Item -Path "backups\stage_f_integration_backup\predict.py.bak" -Destination "src\predict.py" -Force
   Copy-Item -Path "backups\stage_f_integration_backup\app.py.bak" -Destination "app.py" -Force
   ```
   This restores `src/predict.py` and `app.py` to their exact byte-for-byte state prior to Stage F.

---

## 8. Milestone Conclusion and Hold State

- Controlled integration implementation and testing are **COMPLETE**.
- Feature flag `ENABLE_NORMAL_HAND_DETECTOR` is set to **`False`** (disabled).
- All 47 automated regression tests are passing.
- **STOPPING WORK:** Awaiting explicit user approval before enabling the feature in the live application.
