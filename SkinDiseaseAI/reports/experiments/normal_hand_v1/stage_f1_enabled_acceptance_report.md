# Stage F.1: Enabled-Mode Acceptance Test Report

**Date:** October 29, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Target System:** SkinDiseaseAI Streamlit Application & Inference Pipeline  
**Model Checkpoints:**
- Primary 7-Class Dermoscopic Classifier: `models/best_model.keras` (SHA-256: `c6af0869...`, protected & untouched)
- Normal-Hand Appearance Baseline: `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` (SHA-256: `5b2b5345...`, protected & untouched)
- Dermoscopy OOD Gate: `models/ood_gate.npz` (SHA-256: `dcd9974a...`, protected & untouched)

**Status:** Acceptance Test Complete. Feature flag is **RESTORED TO DISABLED (`ENABLE_NORMAL_HAND_DETECTOR = False`)**. No live application activation has taken place. System is stopped awaiting explicit user approval.

---

## 1. Compliance with Strict Execution Requirements

All 6 strict requirements specified for Stage F.1 were enforced throughout execution:

1. **Pre-Test Inspection & Verified Rollback Copies:**
   - Pre-change backups preserved in `backups/stage_f1_acceptance_backup/`:
     - `predict.py.f1_baseline`
     - `app.py.f1_baseline`
   - Pre-existing Stage F backups under `backups/stage_f_integration_backup/` remain sealed and intact.
2. **Controlled Test Scope:**
   - Only the feature flag `ENABLE_NORMAL_HAND_DETECTOR` was modified during the active test execution.
   - The production/demo Streamlit web server was NOT launched or activated.
3. **Zero Modification to Protected Assets:**
   - Core model weights (`best_model.keras`, `isic_finetuned.keras`, `isic_augmented_v1.keras`, `ood_gate.npz`) were untouched.
   - Fixed decision threshold remained locked at **`t = 8.768`** (no retuning or alteration).
   - Training configurations, partition manifests, and the locked backup archive (`SkinDiseaseAI_v1_20261027.zip`) were not modified.
4. **Quarantine Cohort Strictly Protected:**
   - All 243 images from quarantined participants (`27`, `570`, `1566`, and `1000000`) were excluded from test selection, scoring, and feature extraction.
5. **Responsible Research Framing:**
   - This experiment is explicitly documented as an engineering integration acceptance test of an experimental computer vision baseline, **NOT as clinical validation or medical diagnosis**.
6. **Restoration & Safe Hold:**
   - After completing the enabled-mode tests, `ENABLE_NORMAL_HAND_DETECTOR` was restored to **`False`**.
   - The restored state was independently verified via automated assertions and the full 47-test regression suite.

---

## 2. Protected Asset Hash Verifications

All 7 protected assets were re-hashed after testing and verified against the locked pre-experiment backup (`SkinDiseaseAI_v1_20261027.zip`):

| Asset Path | Expected Hash (Locked Backup) | Post-Test Verified Hash | Status |
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

## 3. Enabled-Mode Acceptance Test Suite: Execution & Results

Controlled execution was conducted using `predict_bytes()` and `app.show_result()` with `ENABLE_NORMAL_HAND_DETECTOR = True`:

### Test Case Matrix

| # | Modality / Scenario | Input Identifier & Metadata | Active Routing | Actual Output Fields | Displayed Disclaimers & Limitations | Acceptance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Typical Hand Photograph** | `Hand_0000002.jpg`<br>• Partition: `test`<br>• Subject: `0`<br>• `accessories=0`<br>• `nail_polish=0`<br>• `irregularities=0` | MediaPipe localization $\rightarrow$ Crop $(1124 \times 933)$ $\rightarrow$ MobileNetV3Small $(576\text{-dim}) \rightarrow$ PCA-Mahalanobis | • `ok: True`<br>• `is_hand: True`<br>• `code: None`<br>• `scores: []`<br>• `melanoma_flag: False`<br>• `hand_analysis.decision: USABLE_HAND_IMAGE`<br>• `score: 5.520` (thr: $8.768$)<br>• 7-Class Model: **Suppressed** | *"Image appearance is consistent with reference normal hand skin dataset (score: 5.52 <= threshold 8.77). Notice: A typical visual appearance does not guarantee medical health."*<br>Section 4 Notice: *"DOES NOT certify skin is healthy or disease-free."* | **PASS** |
| **2** | **Hand with Accessories / Jewellery** | `Hand_0000021.jpg`<br>• Partition: `test`<br>• Subject: `0`<br>• `accessories=1`<br>• `nail_polish=0` | MediaPipe localization $\rightarrow$ Crop $(941 \times 893)$ $\rightarrow$ MobileNetV3Small $\rightarrow$ PCA-Mahalanobis | • `ok: True`<br>• `is_hand: True`<br>• `code: None`<br>• `scores: []`<br>• `melanoma_flag: False`<br>• `hand_analysis.decision: USABLE_HAND_IMAGE`<br>• `score: 4.793` (thr: $8.768$)<br>• 7-Class Model: **Suppressed** | Confounder Card: *"Wearing rings, wristwatches, or bracelets roughly quadrupled false anomaly alerts (19.16% with accessories vs. 4.35% without)... While removing accessories reduces false flags, doing so does NOT guarantee a correct result."* | **PASS** |
| **3** | **Unusual / Out-of-Domain Hand** | `webcam_classroom_hand.jpg`<br>• Classroom lighting<br>• Complex desk background | MediaPipe localization $\rightarrow$ Crop $(276 \times 199)$ $\rightarrow$ MobileNetV3Small $\rightarrow$ PCA-Mahalanobis | • `ok: True`<br>• `is_hand: True`<br>• `code: None`<br>• `scores: []`<br>• `melanoma_flag: False`<br>• `hand_analysis.decision: UNUSUAL_RELATIVE_TO_TRAINING_DATA`<br>• `score: 15.700` (thr: $8.768$)<br>• 7-Class Model: **Suppressed** | *"Appearance differs from reference normal hand dataset (score: 15.70 > threshold 8.77). This visual variation may be caused by a lesion, rash, unusual lighting, angle, or foreign object. Notice: This is an experimental visual check, NOT a medical diagnosis."* | **PASS** |
| **4** | **Non-Hand Image** | `tests/fixtures/gaming_poster.jpg`<br>• High-contrast graphic artwork | Cropper finds 0 hands $\rightarrow$ Falls through to dermoscopy domain gate ($d > 2,673.1$) $\rightarrow$ Rejected | • `ok: False`<br>• `is_hand: False`<br>• `code: None`<br>• `scores: []`<br>• `melanoma_flag: False`<br>• `stage_failed: domain_gate`<br>• `hand_analysis: None` | Rejection banner: *"No condition label shown. This photo is outside the validated domain (dermoscopy)."* | **PASS** |
| **5** | **Corrupt / Unreadable Image** | `b"corrupted_image_payload_not_valid_jpeg_or_png"` | `cv2.imdecode` returns `None` $\rightarrow$ Falls through to `check_image` $\rightarrow$ Format check fails | • `ok: False`<br>• `is_hand: False`<br>• `code: None`<br>• `scores: []`<br>• `melanoma_flag: False`<br>• `stage_failed: domain_gate`<br>• `hand_analysis: None` | Rejection banner: *"No prediction is shown. The file is not a readable image."* | **PASS** |
| **6** | **Supported Dermoscopic Close-up** | `ISIC_0029068.jpg`<br>• HAM10000 validation split<br>• Diagnosis: `bkl` (histopathology) | MediaPipe detects 0 hands $\rightarrow$ Falls through to dermoscopy domain gate $\rightarrow$ Passes quality/OOD $\rightarrow$ 7-class classifier | • `ok: True`<br>• `is_hand: False`<br>• `code: "bcc"`<br>• `score: 53.1%`<br>• `scores count: 7`<br>• `melanoma_flag: False`<br>• `hand_analysis: None` (Marked N/A) | Full standard disclaimer (`DISCLAIMER`): *"Educational research prototype — NOT a medical device. Trained only on dermoscopic images of seven pigmented-lesion categories... See a qualified clinician for any skin concern."* | **PASS** |

---

## 4. Specific Verification Items Confirmed

1. **Exclusive Pipeline Routing:**
   - Every macro hand photograph (`Hand_0000002.jpg`, `Hand_0000021.jpg`, `webcam_classroom_hand.jpg`) was processed exclusively through `NormalHandPipeline`.
   - None of the hand photographs were evaluated by the dermoscopic lesion classifier.
2. **Absolute Suppression of Diagnostic Labels on Hands:**
   - In all hand tests, `code` was `None`, `scores` table was empty (`[]`), and `melanoma_flag` was `False`.
   - No condition names, probability distributions, or diagnostic claims were produced.
3. **Preservation of Dermoscopic Behavior:**
   - For `ISIC_0029068.jpg`, the 7-class classifier executed without interference, producing identical predictions (`bcc`, `53.1%`, 7 scores).
   - Hand anomaly scoring was completely inactive (`hand_analysis = None`).
4. **Graceful Handling of Failures:**
   - Corrupt bytes and non-hand artwork halted safely without uncaught exceptions or misleading outputs.
5. **Medical Communication Guardrails:**
   - Hand appearance scores are framed strictly as:
     - *"Consistent with Reference Normal Hand Skin Data"* ($d \le 8.768$)
     - *"Unusual Relative to Reference Normal Hand Data"* ($d > 8.768$)
     - *"Unsuitable or Uncertain Input"* (cropping/resolution failures)
   - Disclaimers state clearly that typical visual appearance does **not** certify healthy skin, and unusual appearance does **not** diagnose disease.
6. **Confounder Communication:**
   - Empirical test statistics ($19.16\%$ false alarms with accessories vs. $4.35\%$ without; $22.50\%$ with nail polish vs. $8.95\%$ without) are displayed with the explicit caveat that removing accessories reduces confounders but does **not** guarantee a correct result.

---

## 5. Streamlit Result Rendering Verification

The user interface rendering in [app.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py) was tested across all 5 test outputs using mock Streamlit dispatch. All UI branches executed with **0 errors and 0 unhandled exceptions**:
- **Case A (Hand Detected with `hand_analysis`):** Correctly renders Section 1 (Modality & Quality), Section 2 (Normal Hand Evaluation with score card, threshold $8.768$, and confounder advisory), Section 3 (7-Class Classifier Suppression Card), and Section 4 (Mandatory Clinical Safety Notice).
- **Case B (Rejected/Out-of-Domain):** Correctly renders rejection banners with specific failure reasons.
- **Case C (Dermoscopy Accepted):** Correctly renders primary class metric, uncalibrated score, 7-class probability table, and melanoma follow-up flags.

---

## 6. Restoration and Final Regression Verification

Following the enabled-mode tests:
1. **Feature Toggle Restored:**
   [src/predict.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py#L40-L42) was restored to:
   ```python
   # Feature toggle for normal hand anomaly detector: defaults to False (disabled) as required
   ENABLE_NORMAL_HAND_DETECTOR = False
   ```
2. **Restored State Confirmed:**
   - Verified that `ENABLE_NORMAL_HAND_DETECTOR is False`.
   - Verified fixed threshold remains **`8.76756...`** ($8.768$).
   - Verified all 7 protected SHA-256 hashes match $100\%$.
3. **Full Regression Suite Output:**
   Command:
   ```powershell
   & ".\.venv\Scripts\python.exe" -m unittest discover -s tests
   ```
   Actual output:
   ```
   ----------------------------------------------------------------------
   Ran 47 tests in 19.794s

   OK
   ```
   All 47 tests passed (0 failures, 0 errors).

---

## 7. Milestone Conclusion & Hold State

- The Stage F.1 Enabled-Mode Acceptance Test is **COMPLETE and SUCCESSFUL**.
- All requirements, suppression invariants, and disclaimers function as specified.
- The live application remains **DISABLED (`ENABLE_NORMAL_HAND_DETECTOR = False`)**.
- The quarantined exploratory cohort ($N=243$) remains unaccessed.
- **STOPPED:** Awaiting your explicit approval before enabling the feature in the live application.
