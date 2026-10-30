# Stage I: Existing Model Audit and Accuracy Improvement Plan

**Date:** October 30, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI (Educational Research Prototype — NOT a Medical Device)  
**Workspace:** `d:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI`  
**Milestone:** Stage I — Existing Model Audit and Accuracy Improvement Plan  
**Status:** Planning & Audit Document — **Read-Only Mode Adhered to (Zero Code Modified)**

---

## 1. Actual Repository Inventory

A comprehensive read-only audit of files, directories, checkpoints, and manifests was conducted across the workspace.

### A. Source Code Modules
* [`src/predict.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py): Core inference routing, input validation dispatch, OOD gating, 7-class disease evaluation, and feature flag controls.
  * Verified state: `ENABLE_NORMAL_HAND_DETECTOR = False` (strictly preserved).
* [`app.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py): Streamlit web application interface with Upload and Camera capture tabs.
* [`src/normal_hand/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/):
  * `cropper.py`: MediaPipe-based hand detection and padded bounding box extraction.
  * `feature_extractor.py`: MobileNetV3Small backbone (pretrained on ImageNet, frozen, 576-dim output). Supports `normalize=True/False`.
  * `anomaly_detector.py`: `MahalanobisAnomalyDetector` class supporting PCA projection, shrinkage regularization ($\lambda$), and distance thresholding.
  * `pipeline.py`: `NormalHandPipeline` tying cropper, extractor, and detector together. Currently hardcodes `normalize=False` at line 69.
* [`src/gate.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/gate.py): Out-of-Distribution (OOD) gating for dermoscopic close-ups.
* [`src/hand_validator.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/hand_validator.py): Input validation, blur/brightness checks, skin fraction, and spoof heuristic filters.

### B. Model Checkpoints & Artifacts
* **Seven-Class Dermoscopic Classifiers (Protected):**
  * [`models/best_model.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/best_model.keras) (Baseline EfficientNetB0, SHA-256: `C6AF0869...`, Verified Match with locked zip).
  * [`models/isic_augmented_v1.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_augmented_v1.keras) (Augmented EfficientNetB0, SHA-256: `FE3EB872...`, Verified Match).
  * [`models/isic_finetuned.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_finetuned.keras) (Fine-tuned EfficientNetB0, SHA-256: `6149DBD4...`, Verified Match).
  * [`models/experiments/isic_ens_beta999.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/isic_ens_beta999.keras) (Ensemble candidate model).
* **Domain Gating:**
  * [`models/ood_gate.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/ood_gate.npz) (Dermoscopy OOD Gate, SHA-256: `DCD9974A...`, Verified Match).
* **Hand Appearance Models:**
  * [`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/hand_anomaly_baseline.npz): Stage D Baseline ($k=48$, $\lambda=0.05$, $t=8.768$, raw 576-dim features). SHA-256: `5B2B5345...`.
  * [`models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz): Stage G Winning Candidate ($k=32$, $\lambda=0.05$, $t=6.931$, L2 unit-sphere features). SHA-256: `C4F0FED9...`.
  * `candidate_1a_l2_mahalanobis_k48.npz`, `candidate_3_square_crop_l2_k48.npz` (Stage G alternative candidates).
* **Empty Directories (Verified):**
  * `models/normal_disease/`: **Empty** (0 files).
  * `models/normal_vs_condition/`: **Empty** (0 files).
  * *Confirmation:* There is no trained normal-vs-disease classifier anywhere in the repository.

### C. Datasets & Manifests
* [`data/experiments/normal_hand_v1/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv): 11k Hands dataset manifest (11,076 rows across 189 participants; 129 train, 28 val, 28 test, 4 quarantined).
* `D:\DOWNLOADS D\archive (2)\Oily-Dry-Skin-Types`: 3,152 facial cosmetic skin type images (`dry`, `normal`, `oily`).
* `D:\DOWNLOADS D\archive (4)`: 2,147 clinical images (`eczema`, `tinea`).
* `D:\DOWNLOADS D\archive`: HAM10000 dermoscopy collection (10,015 images).
* `data_clinical/normal_webcam/`: User cohort directory with `people.csv` header. Awaiting consented webcam hand photos.

### D. Test Fixtures & Test Suite
* `tests/fixtures/gaming_poster.jpg` (Anime / non-skin artwork).
* `tests/fixtures/webcam_classroom_hand.jpg` (Webcam hand photo).
* Full regression test suite: 47 automated tests passing (`Ran 47 tests in 9.599s OK`).

---

## 2. Model-to-Task Mapping Table

| Task | Existing Artifact | Training Labels | Supported Input Domain | Verified Status & Clinical Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Invalid / Non-Skin Input Rejection** | `src/hand_validator.py` (`validate_hand_input`) & `src/normal_hand/cropper.py` | Heuristic skin masks, blur/brightness, MediaPipe landmark topology | Uploaded or webcam images | **Verified Operational.** 100% rejection of gaming posters and synthetic noise in regression tests. Does not prove the image is a living human. |
| **Normal-Looking Hand Appearance Modeling** | `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` & `stage_g/candidate_1b_...npz` | Unsupervised / One-class (Fitted exclusively on 129 healthy studio individuals) | Dorsal and palmar hand crops on uniform neutral backgrounds | **Verified Appearance Detector Only.** Measures statistical deviation from healthy studio hands. **NOT a medical disease classifier.** Cannot confirm absence of disease. |
| **Normal Facial Skin Type** | None deployed (Dataset: `archive (2)\Oily-Dry-Skin-Types`) | Cosmetic skin types (`dry`, `normal`, `oily`) | Facial close-ups (foreheads, cheeks, noses) | **Cosmetic Sebum Metric Only.** In cosmetic science, "normal" skin refers strictly to eudermic skin (balanced sebum/moisture). **Has zero medical validity for pathology or cancer screening.** |
| **Actual Skin-Condition Classification** | `models/best_model.keras` + `models/isic_augmented_v1.keras` | 7 pigmented lesion categories (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) | Dermoscopic close-up optical images passing OOD gate | **Verified Dermoscopic Classifier.** Validated strictly on dermoscopy optics. Completely invalid on macro-camera phone hand photos. |

### Critical Boundaries to Enforce:
1. **Cosmetic skin type $\neq$ Absence of disease:** Eudermic ("normal") skin can harbor malignant melanoma or squamous cell carcinoma; conversely, "oily" skin is normal human physiology.
2. **Anomaly detection $\neq$ Disease classification:** A high anomaly distance on a hand photo indicates deviation from a studio-lit white-paper dataset (often caused by rings, watches, nail polish, wrinkles, or room lighting), not verified disease.
3. **Unusual visual appearance $\neq$ Confirmed pathology:** An unusual mark may be a benign freckle, scar, ink stain, or lighting glare.
4. **Normal visual appearance $\neq$ Proof of disease-free skin:** Early amelanotic melanoma, flat warts, or subungual lesions may appear visually unremarkable to standard camera models.

---

## 3. Current Measured Performance and Evidence Source

### A. Seven-Class Dermoscopic Classifier
* **Source:** Evaluated strictly on the preserved frozen HAM10000 test split ($N=1,494$ images, 187 melanomas; `outputs/melanoma/stage3_test_bootstrap_results.json`).
* **Selected Configuration:** Two-model ensemble (Original Baseline + Augmented v1) with validation-calibrated melanoma decision threshold $t = 0.347$.
* **Measured Performance (with Bootstrap 95% Confidence Intervals):**
  * **Test Accuracy:** **`72.29%`** ($1,080 / 1,494$) [95% CI: $70.01\% - 74.50\%$]
  * **Test Melanoma Recall:** **`54.01%`** ($101 / 187$) [95% CI: $47.06\% - 60.96\%$] (Restored from $53.48\%$ baseline)
  * **Test Melanoma Precision:** **`41.22%`** ($101 / 245$) [95% CI: $35.10\% - 47.35\%$]
  * **False Alarms (Non-melanoma flagged as mel):** **`144 / 1,307`** [95% CI: $122 - 167$]
  * **Test Macro F1:** **`0.584`**

### B. Normal Hand Anomaly Baseline (Candidate 0)
* **Source:** Evaluated on held-out test partition ($N=420$ images, 28 unseen individuals; `reports/experiments/normal_hand_v1/stage_d_baseline_report.md`).
* **Calibrated Threshold:** $t = 8.768$ (calibrated on validation split at $\alpha = 0.05$).
* **Measured Performance:**
  * **Overall Test False-Rejection Rate:** **`10.24%`** ($43 / 420$) [Wilson 95% CI: $7.68\% - 13.51\%$]
  * **Clean Hands (No Accessories):** **`4.35%`** ($11 / 253$)
  * **Hands with Accessories:** **`19.16%`** ($32 / 167$) $\rightarrow$ **$4.41\times$ Disparity**
  * **Hands with Nail Polish:** **`22.50%`** ($9 / 40$) $\rightarrow$ **$2.51\times$ Disparity**
  * **Feature Norm Correlation:** **`+0.656`** (Strong confounding by vector magnitude)

---

## 4. Review of the Stage G Improvement: Candidate 1B vs. Baseline

### A. Architectural and Parametric Comparison

| Parameter / Metric | Baseline (Candidate 0) | Candidate 1B (Recommended) | Technical Difference |
| :--- | :--- | :--- | :--- |
| **Saved Artifact Path** | `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` | `models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz` | Independent experimental directory |
| **Artifact SHA-256** | `5B2B53451932ACC9644AB612AA0AF7FF0033E8A03ED4503C8D2EF686DCB8DBF4` | `C4F0FED96CB0308C8155CED51607E0DF872F4C91EC2180ADCD4ED311267FC5AB` | Verified intact |
| **Artifact Size** | $123.7\text{ KB}$ | **$80.2\text{ KB}$** | **$35.2\%$ smaller footprint** |
| **Feature Normalization** | Unnormalized ($\mathbb{R}^{576}$) | **L2 Unit Sphere ($\mathbb{S}^{575}$)** | Eliminates magnitude confounding |
| **PCA Latent Dimensions ($k$)** | $48$ components ($83.87\%$ variance) | **$32$ components ($78.2\%$ variance)** | Prunes noisy, fragile trailing axes |
| **Covariance Shrinkage ($\lambda$)**| $0.05$ | $0.05$ | Ledoit-Wolf-style spherical shrinkage |
| **Validation Calibrated Threshold ($t$)** | `8.768` | **`6.931`** | Nominal $\alpha = 0.05$ calibration |

### B. Empirically Measured Improvements (Stage G Evaluation)
* **Overall Test False Alerts:** Dropped from **`10.24%`** to **`7.86%`** ($33 / 420$)—a **`23.2%` relative reduction**.
* **Accessory Subgroup False Alerts:** Dropped from **`19.16%`** to **`13.17%`** ($22 / 167$)—a **`31.3%` relative reduction**.
* **Accessory Disparity Ratio:** Decreased from **$4.41\times$ to $3.03\times$**.
* **Clean Hands False Alerts:** Remained identical at **`4.35%`** ($11 / 253$)—no penalty on bare hands.
* **Norm Confounding Correlation:** Decreased from **`+0.656`** down to **`+0.285`** (**$56.6\%$ reduction**).
* **Non-Hand Discrimination:** Maintained **$100\%$ rejection** on non-hand inputs (`gaming_poster.jpg` and Gaussian noise).
* **Inference Speed:** CPU scoring time reduced from $0.21\text{ ms}$ to **$0.14\text{ ms}$**.

---

## 5. Candidate 1B Integration Readiness Assessment

| Requirement | Current Status | Blocker / Action Needed |
| :--- | :---: | :--- |
| **Model Weights & Matrices** | Ready | `candidate_1b_l2_mahalanobis_k32.npz` is fully saved and validated. |
| **Feature Normalization Pipeline** | **Not Ready** | `src/normal_hand/pipeline.py` (line 69) currently hardcodes `normalize=False`. Must accept `normalize=True`. |
| **Inference Threshold** | Ready | Calibrated validation threshold is $6.931$ (stored in the `.npz` header). |
| **Application Wiring (`predict.py`)** | Ready | Currently loads baseline via `get_normal_hand_pipeline()`. Needs parameter to load Candidate 1B. |
| **Feature Flag Safety** | Intact | `ENABLE_NORMAL_HAND_DETECTOR = False` is strictly enforced. |

**Verdict:** Candidate 1B is **computationally and mathematically ready**, but requires a 2-line configuration update in `src/normal_hand/pipeline.py` and `src/predict.py` before activation.

---

## 6. Identified Accuracy Limitations & Root Causes

1. **Absence of True Binary Normal-vs-Condition Hand Dataset:**
   * The repository possesses only normal studio hands (11k Hands) and non-hand clinical body images (`archive (4)`).
   * **Root cause:** No paired, verified hand dermatosis dataset exists under consumer lighting. Any claim of a "disease classifier for hands" would be fraudulent.
2. **Feature Norm Inflation on Metallic / Reflective Surfaces:**
   * In raw MobileNetV3 features, reflective jewelry (rings, watches) and glossy nail polish produce unusually high activation magnitudes.
   * **Root cause:** Resolved in Candidate 1B by L2 unit-sphere normalization, but nail polish still exhibits residual disparity ($20.0\%$ FPR vs $6.58\%$ unpolished) due to color/texture differences.
3. **Background & Domain Confounding:**
   * 11k Hands images were taken on uniform white paper under studio lamps.
   * Real-world phone photos have varied backgrounds (wood, desks, classroom walls) and ambient lighting.
   * **Root cause:** MediaPipe hand crop isolates the hand, but bounding-box corners still include background pixels.
4. **Cosmetic Label Misattribution:**
   * Misinterpreting the `Oily-Dry-Skin-Types` dataset as "healthy vs diseased" skin.
   * **Root cause:** Terminology confusion between cosmetic typology ("normal" = eudermic) and pathology ("normal" = non-pathological).

---

## 7. Proposed Improvements Ranked by Expected Benefit and Evidence

| Rank | Proposed Improvement | Expected Benefit | Level of Evidence | Risk Level |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **Upgrade Normal Hand Detector to Candidate 1B** ($k=32$, L2-normalized, $t=6.931$) | Reduces false anomaly alarms on hands by $23.2\%$ overall and $31.3\%$ on jewelry. | **High** (Measured on 420 unseen test images across 28 individuals). | Very Low (Reversible drop-in replacement). |
| **2** | **Implement Strict Stage 1 Suitability Gate** (HSV/YCrCb skin area fraction $> 15\%$ + decodability + blur) | Rejects cars, anime, gaming posters, and corrupt inputs before downstream inference. | **High** (Verified on 20 non-hand and noise fixtures). | Zero (Purely protective rejection). |
| **3** | **Enforce Standardized Three-Outcome Wording for Hand Photos** | Prevents false reassurance and false disease panic; eliminates "Good" or "disease detected". | **Absolute** (Ethical & regulatory compliance for prototype). | Zero (UI display only). |
| **4** | **Skin-Patch / Palm-Only ROI Extraction** (Exclude wrist and fingertips) | Eliminates wristwatches and fingernail polish confounders entirely. | **Medium** (Hypothesized based on Stage D error analysis). | Low (Requires MediaPipe landmark segmentation). |
| **5** | **Acquire Real Consented Clinical Hand Dataset** (`data_clinical/`) | Enables true supervised normal-vs-condition hand classification. | **Fundamental** (Required for future clinical utility). | Dependent on data collection. |

---

## 8. Additional Data Requirements to Close the Gap

To advance beyond one-class appearance modeling to a validated normal-vs-condition hand classifier:
1. **Normal Webcam Hands (`data_clinical/normal_webcam`):**
   * 50–100 consented individuals across Fitzpatrick skin tones I–VI.
   * Captured using phone cameras and webcams under ambient daylight and indoor lighting.
   * Metadata recorded in `people.csv` (no facial photographs).
2. **Clinically Verified Hand Lesion Dataset:**
   * 200–500 photographs of verified common hand conditions (e.g. dyshidrotic eczema, contact dermatitis, plaque psoriasis, tinea manuum, verruca vulgaris).
   * Captured in natural macroscopic lighting (not dermoscopy).
3. **Partition Discipline:**
   * Strict person-disjoint splits (70% train, 15% validation, 15% test). Never evaluate on the same individual across splits.

---

## 9. Future Source Files Affected & Rollback Strategy

If approved in a future implementation stage, only **two files** will require minor modifications:

### A. [`src/normal_hand/pipeline.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/pipeline.py)
* **Proposed change:** Add `normalize: bool = False` parameter to `NormalHandPipeline.__init__` and pass it to `extract_features(crop_res.cropped_bgr, normalize=self.normalize)`.
* **Rationale:** Allows pipeline to execute Candidate 1B's L2 unit-sphere projection instead of hardcoded raw embeddings.
* **Reversibility:** Fully backwards compatible (default `False`).

### B. [`src/predict.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py)
* **Proposed change:** Update `get_normal_hand_pipeline()` to load `candidate_1b_l2_mahalanobis_k32.npz` with `normalize=True` when `ENABLE_NORMAL_HAND_DETECTOR` is enabled.
* **Rationale:** Activates the superior Stage G model without modifying any baseline files.
* **Reversibility:** Instant rollback by pointing back to `hand_anomaly_baseline.npz`.

### C. Rollback Safeguard
* Existing verified backups:
  * `backups/stage_f_integration_backup/`
  * `backups/stage_f1_acceptance_backup/`
* Protected backup archive: `D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip` (100% verified intact).

---

## 10. Clear Distinction: Verified Results vs. Hypotheses

| Category | Claim / Finding | Status | Supporting Evidence |
| :--- | :--- | :---: | :--- |
| **Verified** | Protected assets remain 100% identical to locked zip archive. | **VERIFIED** | SHA-256 hash comparison across all 7 checkpoints. |
| **Verified** | Seven-class dermoscopic ensemble achieves $72.29\%$ accuracy and $54.01\%$ melanoma recall on frozen test set. | **VERIFIED** | Bootstrap 95% CIs evaluated on 1,494 HAM10000 images. |
| **Verified** | Candidate 1B reduces test false alarms from $10.24\%$ to $7.86\%$ and accessory disparity from $4.41\times$ to $3.03\times$. | **VERIFIED** | Measured post-hoc on 420 test images across 28 unseen individuals. |
| **Verified** | `archive (2)` is a cosmetic facial dataset and cannot classify hand skin diseases. | **VERIFIED** | Dataset content audit (3,152 facial photos with cosmetic sebum labels). |
| **Verified** | No trained normal-vs-disease classifier exists in `models/normal_disease/` or `models/normal_vs_condition/`. | **VERIFIED** | Directory inspection confirmed folders are empty. |
| **Hypothesis** | Cropping only the palm or dorsal skin patch will completely remove the remaining $13.17\%$ accessory false alert rate. | **HYPOTHESIS** | Requires empirical landmark segmentation experiment. |
| **Hypothesis** | Training an EfficientNetB0 on phone webcam hands will achieve $>80\%$ sensitivity on hand dermatitis. | **HYPOTHESIS** | Requires acquiring the verified clinical hand condition dataset. |

---

## 11. Final Stop Condition & Recommendation

### Recommendation for Single Best Next Step:
**Stage I audit and planning is complete.**
The single best next step is:
1. Retain `ENABLE_NORMAL_HAND_DETECTOR = False` until full review.
2. In the next stage (Stage J), perform the controlled, reversible update to `src/normal_hand/pipeline.py` and `src/predict.py` to point to **Candidate 1B** with `normalize=True`, run the full regression test suite (47 tests), and evaluate against the real test fixtures.

**STOP.** Awaiting user written approval before writing any integration code or modifying any application files.
