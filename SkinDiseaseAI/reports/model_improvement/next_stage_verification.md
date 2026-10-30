# 🔍 SkinDiseaseAI — Stage 1: Read-Only Implementation Verification Report

**Project:** SkinDiseaseAI — Read-Only Implementation & Software Quality Audit  
**Role:** Senior Machine Learning Engineer, Deep Learning Researcher & Software Quality Auditor  
**Date & Timestamp:** October 29, 2026 | 14:22 IST (Local Time)  
**Execution Context:** Native Windows 11 | Python 3.12.10 | TensorFlow 2.21.0 | 24 GB RAM | NVIDIA RTX 3050 (CPU execution via oneDNN/AVX2)  
**Audit Mode:** Strict Read-Only (Zero code edits, zero model retraining, zero threshold modifications, zero file overwrites)  

---

## 1. Audit Scope & Evidence Standards

This audit evaluates the actual, runnable implementation of the **SkinDiseaseAI** codebase against historical reports and architectural specifications. 

To maintain strict scientific and software quality standards, every technical claim is explicitly categorized:
- **[Verified from source code]**: Confirmed by inspecting physical lines in the source files.
- **[Verified by executed test]**: Validated by running a non-destructive unit/integration test.
- **[Reported by existing documentation]**: Recorded from historical JSON/CSV reports without active re-execution.
- **[Not verified / evidence unavailable]**: Flagged where physical evidence or labelled data is absent.

---

## 2. Files Actually Inspected

The following physical files were examined during this audit:

1. **Application & Inference Source Code:**
   - [`app.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py) (Lines 1–315): Streamlit frontend, page routing, model selector, pipeline stepper.
   - [`src/predict.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py) (Lines 1–240): End-to-end inference pipeline, ensemble averaging, melanoma follow-up thresholding, early abstention.
   - [`src/preprocessing.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/preprocessing.py) (Lines 1–141): Image decoding, resizing, pixel scaling (`[0, 255]`), online augmentation layer.
   - [`src/dataset.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/dataset.py) (Lines 1–133): Dataset discovery, lesion-level grouping, class definitions.
   - [`src/gate.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/gate.py) (Lines 1–149): Mahalanobis OOD gate, quality heuristics (blur, luminance).
   - [`src/hand_validator.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/hand_validator.py) (Lines 1–380): Skin-pixel color screening, MediaPipe landmarking, anti-spoofing screen.
2. **Configuration & Indices:**
   - [`config.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/config.json) (Lines 1–51): Core project paths, hyperparameters, calibrated melanoma threshold block.
   - [`models/class_indices.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/class_indices.json) (Lines 1–9): Class name to numerical index dictionary.
3. **Existing Model-Improvement Reports:**
   - [`reports/model_improvement/project_audit.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/model_improvement/project_audit.md)
   - [`reports/model_improvement/baseline_metrics.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/model_improvement/baseline_metrics.json)
   - [`reports/model_improvement/seven_class_improvement_results.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/model_improvement/seven_class_improvement_results.md)
   - [`reports/model_improvement/normal_vs_disease_readiness.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/model_improvement/normal_vs_disease_readiness.md)
   - [`reports/model_improvement/smartphone_generalization_plan.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/model_improvement/smartphone_generalization_plan.md)
   - [`reports/model_improvement/final_validation_report.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/model_improvement/final_validation_report.md)
4. **Test Suite:**
   - [`tests/test_gate_quality.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/test_gate_quality.py)
   - [`tests/test_augmentation_and_weights.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/test_augmentation_and_weights.py)
   - [`tests/test_hand_validation_and_spoof.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/test_hand_validation_and_spoof.py)
   - [`tests/test_regression_fixtures.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/test_regression_fixtures.py)

---

## 3. Protected Asset Verification & Cryptographic Integrity

All eight protected assets were inspected on disk. SHA-256 digests and file sizes were computed via scratch script and cross-referenced against historical audit records:

| Protected File Path | Physical Existence | Measured Size | Last Modified | Measured SHA-256 Digest | Audit Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models\best_model.keras` | **EXISTS** | 33,486,076 B | 2026-10-15 21:50:51 | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **[Verified by executed test]** Intact |
| `models\isic_finetuned.keras` | **EXISTS** | 33,477,534 B | 2026-10-23 00:39:24 | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **[Verified by executed test]** Intact |
| `models\isic_augmented_v1.keras` | **EXISTS** | 33,477,861 B | 2026-10-26 12:18:27 | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **[Verified by executed test]** Intact |
| `models\ood_gate.npz` | **EXISTS** | 6,102,303 B | 2026-10-26 10:30:41 | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **[Verified by executed test]** Intact |
| `outputs\splits.csv` | **EXISTS** | 1,259,807 B | 2026-10-15 21:23:43 | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **[Verified by executed test]** Intact |
| `outputs\combined_splits.csv` | **EXISTS** | 2,556,961 B | 2026-10-23 00:11:18 | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **[Verified by executed test]** Intact |
| `config.json` | **EXISTS** | 1,985 B | 2026-10-26 15:46:08 | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **[Verified by executed test]** Intact |
| `D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip` | **EXISTS** | 1,506,237,677 B | 2026-10-27 10:10:21 | `11badbdd6b32140e55660c7ea2d3a499a3ba197ea222d29cb06b5bdaabe9bacd` | **[Verified by executed test]** Intact |

**Conclusion:** Zero protected assets have been altered, overwritten, or corrupted.

---

## 4. Application & Inference Pipeline Audit

### A. Model Loading & Ensemble Configuration
- **Model Files Loaded:**
  - In [`app.py:25-36`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py#L25-L36), `available_models` checks `OLD_MODEL` (`models/best_model.keras`) and `AUG_MODEL` (`models/isic_augmented_v1.keras`). **[Verified from source code]**
  - The dictionary maps `"Ensemble: Original + Aug v1 (Calibrated Follow-up)"` $\to$ `"ensemble"`. Because this entry is inserted first, `st.sidebar.selectbox` defaults to the Ensemble. **[Verified from source code]**
  - In [`src/predict.py:155-164`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py#L155-L164), when `model_path == "ensemble"`, both `best_model.keras` and `isic_augmented_v1.keras` are loaded via `load_artifacts`. Their outputs are averaged: `probs = 0.5 * (p_orig + p_aug)`. **[Verified from source code]**
- **Class Ordering & Indices:**
  - Both models output 7 units matching [`models/class_indices.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/class_indices.json):
    `{akiec: 0, bcc: 1, bkl: 2, df: 3, mel: 4, nv: 5, vasc: 6}`. **[Verified from source code]**
- **Dimensions & Scaling:**
  - Input dimensions are verified as `(224, 224, 3)`. Both training and inference use `float32` arrays in `[0.0, 255.0]`. EfficientNetB0 handles normalization internally without double-rescaling. **[Verified from source code]**

### B. Melanoma Threshold & Follow-Up Flag
- **Configured Threshold:**
  - In [`config.json:38-48`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/config.json#L38-L48), the ensemble melanoma threshold is explicitly set to `0.3466` (rounded to approximately `0.347`). **[Verified from source code]**
- **Application in Inference:**
  - In [`src/predict.py:179-188`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py#L179-L188), the code extracts `mel_prob = float(probs[mel_idx])` from the ensemble-averaged probability vector.
  - The threshold check `if mel_prob >= t_calibrated` operates directly on the ensemble probability. **[Verified from source code]**
- **Reported Test Metrics for Threshold `0.3466`:**
  - In [`outputs/melanoma/stage3_test_bootstrap_results.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/melanoma/stage3_test_bootstrap_results.json):
    - Mel Recall: `54.01%` (101/187, 95% CI: `[47.2%, 60.8%]`). **[Reported by existing documentation]**
    - Mel Precision: `41.74%` (95% CI: `[35.8%, 47.9%]`). **[Reported by existing documentation]**
    - Non-Melanoma False Alarms: `141 / 1,307` (`10.79%`, 95% CI: `[120, 165]`). **[Reported by existing documentation]**
    - Unflagged Melanomas: `86 / 187` (`45.99%` of test melanomas were not flagged). **[Reported by existing documentation]**

### C. Input Suitability, Rejection & Abstention
- **Gate Structure:**
  - In [`src/predict.py:84-111`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py#L84-L111), incoming bytes pass through `check_image(data)` (Domain Gate) unless validation mode is set to `"bypass"`. **[Verified from source code]**
  - If `gate_verdict["status"] == "rejected"`, `predict_bytes` returns early with `ok = False` and `stage_failed = "domain_gate"`. **[Verified from source code]**
  - In [`app.py:128-168`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py#L128-L168), when `result["ok"] == False`:
    - The UI displays an error block (`🛑 Input Rejected at Stage: ...`).
    - The disease classifier is **blocked** (`⛔ Blocked`).
    - The code issues an early `return`. **Zero classification labels, scores, tables, or melanoma flags are rendered.** **[Verified from source code]**
- **Vulnerability Audit on Unsuitable Images:**
  - *Non-skin objects (vehicles, artwork, textures):* Rejected at Stage 1 (Domain Gate) or Stage 2 (Skin Pixels $\le 10\%$). **[Verified by executed test]**
  - *Screen Spoofs / Printed Photos:* Screened by anti-spoofing filter (moiré / bezel checks). **[Verified by executed test]**
  - *Smartphone photo of an arm:* The Mahalanobis distance ($d \approx 5755$) exceeds `2673.08`, so it is **rejected at Stage 1 (Domain Gate)** when in `all_skin` mode. It does NOT pass through to classification unless the user manually selects `"Bypass input validation"`. **[Verified from source code]**
- **Data Foundation of Gate Thresholds:**
  - In `models/ood_gate.npz`, metadata confirms: `{"n_fit": 4000, "n_thr": 1500, "accept_pct": 97.0, "fitted_on": "train", "thresholds_chosen_on": "validation"}`.
  - The gate parameters were fitted strictly on **dermoscopy images**. **[Verified from source code]**
  - The non-hand benchmark matrix in `tests/test_hand_validation_and_spoof.py` is a **synthetic smoke test**, NOT a real-world measured rate. **[Verified from source code]**

### D. Safety Messages & Medical Disclaimers
- In [`src/predict.py:31-33`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py#L31-L33) and [`app.py:18`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py#L18), the prominent banner reads:
  *"Educational research prototype - NOT a medical device. It was trained only on dermoscopic images of seven pigmented-lesion categories and cannot tell whether skin is healthy or diagnose any condition. See a qualified clinician for any skin concern."* **[Verified from source code]**
- In [`app.py:190-193`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py#L190-L193), the melanoma banner explicitly warns:
  *"No flag does NOT mean no melanoma. About 46% of melanomas in testing were not flagged. Notice: This is an exploratory follow-up screening flag, NOT a medical diagnosis. Seek formal evaluation by a board-certified dermatologist."* **[Verified from source code]**
- The UI does **not** state that a user is disease-free or that benign lesions represent normal skin. **[Verified from source code]**

---

## 5. Non-Destructive Test Execution Results

The unit and integration test suite was executed using the project's virtual environment:
- **Command:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -t .`
- **Execution Time:** 5.884 seconds
- **Results:** **33 tests run: 32 passed, 0 failed, 1 skipped.** **[Verified by executed test]**
- **Skipped Test:** `test_fixtures_exist_and_rejected` in `tests/test_regression_fixtures.py` skipped because the directory `tests/fixtures/` does not yet exist.
- **Side Effects:** Zero files created, deleted, or altered.

---

## 6. Confirmed Discrepancies Between Reports & Implementation

The audit identified the following specific discrepancies:

1. **Report Filename Prefixes:**
   - *Reported in prompt:* `reports/model_improvement/1. project_audit.md`, `reports/model_improvement/2. baseline_metrics.json`, etc.
   - *Actual on disk:* Files are named `project_audit.md`, `baseline_metrics.json`, etc. without numeric prefixes. **[Verified from source code]**
2. **Threshold Value Representation:**
   - *Reported in text:* Threshold reported as "approximately `0.347`".
   - *Actual in config:* [`config.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/config.json#L39) defines exact float `0.3466`. **[Verified from source code]**
3. **Domain Gate Operational Scope:**
   - *Existing documentation note:* The domain gate threshold (`2673.08`) was designed to reject non-dermoscopic data.
   - *Implementation Reality:* In `app.py`, selecting "Dermoscopy close-up (validated domain)" enforces `check_image()`. Consequently, **genuine smartphone skin photos are currently blocked at Stage 1**. Users can only pass smartphone photos if they select "Bypass input validation", which removes safety boundaries. **[Verified from source code]**
4. **Regression Fixtures Missing:**
   - *Implementation Expectation:* `tests/test_regression_fixtures.py` expects real webcam arm and gaming poster images in `tests/fixtures/`.
   - *Actual on disk:* `tests/fixtures/` has not been provided by the user, causing the regression test to skip. **[Verified from source code]**

---

## 7. Smartphone Generalization Readiness

- **Current Status:** **NOT PRODUCTION READY FOR SMARTPHONE PHOTOGRAPHY.** **[Verified from source code]**
- **Evidence:**
  - The Mahalanobis gate parameters (`mu`, `prec`, `thr = 2673.08`) in `models/ood_gate.npz` were fitted strictly on 4,000 dermoscopic training images. **[Verified from source code]**
  - Smartphone skin images produce distances $d \approx 5755 \gg 2673.08$, triggering immediate false rejection. **[Reported by existing documentation]**
- **Target Metrics:**
  - Target False Rejection Rate (FRR) $\le 5\%$ and False Acceptance Rate (FAR) $\le 10\%$ are **unvalidated targets**, not measured achievements. **[Reported by existing documentation]**
- **Required Data to Achieve Validated Smartphone Readiness:**
  - At least 400+ genuine smartphone skin photos across Fitzpatrick Skin Types 1–6 under daylight, warm incandescent, and cool fluorescent lighting.
  - At least 200+ real-world non-skin negative images (clothing, interior rooms, furniture, documents, pets).

---

## 8. Normal vs. Disease Classifier Status

> [!CAUTION]
> **BINARY CLASSIFIER STATUS: STRICTLY BLOCKED**  
> Physical inspection confirms that **0 verified Class 0 (normal skin) images exist on disk**. **[Verified from source code]**
> 
> - Neither HAM10000 nor ISIC 2019 contains verified normal skin controls.
> - Stanford DDI contains 171 malignant and 485 benign lesions, but **0 normal skin controls**.
> - Google SCIN contains crowdsourced clinical complaints, not healthy controls.
> - **Prohibition:** Benign lesions (moles, warts, cysts) are pathological abnormalities and must never be substituted as Class 0.
> - **Unblocking Requirement:** The Option A volunteer collection (50 consenting subjects under informed consent) must be executed before binary classifier training can begin.

---

## 9. Risks, Missing Evidence & Recommended Next Action

### Key Risks
1. **Smartphone Misclassification Risk:** Running the 7-class dermoscopic model on smartphone photos without the domain gate exposes users to out-of-domain hallucinations.
2. **Melanoma Under-Diagnosis Risk:** At default argmax, `isic_augmented_v1` missed 62% of melanomas. While the calibrated ensemble ($t=0.347$) improves sensitivity to 54%, ~46% of melanomas remain unflagged on the test set.
3. **Ultra-Low Support on Rare Classes:** `df` has only 7 test samples and `vasc` has 21. Metrics on these classes have wide confidence intervals.

### Recommended Next Action
1. **Do NOT start binary classifier training** until Class 0 volunteer collection is completed.
2. **Do NOT recalibrate the gate** until real smartphone skin photos and non-skin negative images are provided.
3. **Maintain the Calibrated Ensemble as default** in `app.py` for dermoscopic evaluation.
4. **STOP and wait for user review and explicit instructions.**
