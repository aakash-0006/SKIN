# Stage H: Repository Audit — Two-Stage Skin Recognition & Dataset Inventory

**Date:** October 30, 2026  
**Auditor:** Machine Learning Systems Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI (Educational Research Prototype — NOT a Medical Device)  
**Workspace:** `d:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI`  
**Milestone:** Stage H — Two-Stage Workflow Architecture Repository Audit  
**Status:** Audit Completed — **Awaiting User Approval Before Integration Code**

---

## 1. Executive Summary

This audit assesses the existing repository state to prepare for the integration of the intended two-stage skin recognition workflow:
1. **Stage 1 (Input Validation & Suitability):** Fast rejection of corrupted, unreadable, blank, and non-skin inputs (cars, anime, gaming posters, unrelated objects) using image quality and skin/hand localization checks.
2. **Stage 2 (Normal / Abnormal Skin Routing):** Assessment of hand and macroscopic clinical skin inputs.
3. **Stage 3 (Existing Disease Evaluation):** Dermoscopic close-up evaluation preserved strictly for compatible dermoscopic lesions on the 7-class path (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

### Key Findings of the Audit:
1. **100% Bit-for-Bit Integrity of Protected Assets:** Every protected `.keras` model, OOD gate, dataset split manifest, and configuration was verified against the locked backup archive (`D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip`) and confirmed identical.
2. **No Validated Normal-vs-Condition Model Exists in the Repository:**
   - Folders `models/normal_disease` and `models/normal_vs_condition` are currently **empty**.
   - The Stage D/G hand model (`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`) is **strictly a one-class appearance density/anomaly detector** trained exclusively on healthy hands in studio settings. It was **never trained on or evaluated against abnormal hands**, and **cannot** be described or used as a normal-vs-abnormal disease classifier.
3. **The "Oily-Skin" Dataset is Cosmetic Facial Data, Not a Medical Classifier:**
   - The `Oily-Dry-Skin-Types` dataset (`archive (2)`) consists of **facial close-ups** categorized by cosmetic sebum levels (`dry`, `normal`, `oily`).
   - In cosmetic dermatology, "normal" skin refers strictly to **eudermic skin** (balanced hydration and sebum, neither dry nor oily). It does **NOT** denote absence of disease, infection, or malignancy. It cannot be used as a disease/normal classifier.
4. **Governing Policy Implementation:**
   - Per the project rules: **Never display "Good", "healthy", "no disease" or "disease-free"**. The required wording for validated normal appearance is strictly:
     > *"Looks normal in this photo (no visible abnormality detected). Not a medical clearance."*
   - Since no model exists that can reliably distinguish normal hand skin from hand conditions, hand photos must be routed to:
     > *"Hand detected. This system cannot yet tell normal skin from a skin condition."*
   - `ENABLE_NORMAL_HAND_DETECTOR` remains **`False`** until explicit approval.

---

## 2. Protected Asset Hash Verification vs. Locked Backup

Archive: `D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip`  
All 7 protected assets were extracted to an isolated memory buffer and their SHA-256 hashes compared against active repository files:

| Protected File Path | Archive SHA-256 | Live SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `models/best_model.keras` | `C6AF0869563F708D1DC26E67668F0838D7EDFD277B7B585FA65527BCFFDDF2FD` | `C6AF0869563F708D1DC26E67668F0838D7EDFD277B7B585FA65527BCFFDDF2FD` | **100% Match** |
| `models/isic_finetuned.keras` | `6149DBD49D5DC403CF81E55BA118B4CD20D70D4F7A0098617BAE994962005FF2` | `6149DBD49D5DC403CF81E55BA118B4CD20D70D4F7A0098617BAE994962005FF2` | **100% Match** |
| `models/isic_augmented_v1.keras` | `FE3EB872F919BCDAA4AE69C318993B50A33C0FEE815D00AADD2BAB75ADA99D34` | `FE3EB872F919BCDAA4AE69C318993B50A33C0FEE815D00AADD2BAB75ADA99D34` | **100% Match** |
| `models/ood_gate.npz` | `DCD9974A82E40E4DECFB1170893B7934C9BAB9B05C9DF316046F25EA47479DBE` | `DCD9974A82E40E4DECFB1170893B7934C9BAB9B05C9DF316046F25EA47479DBE` | **100% Match** |
| `outputs/splits.csv` | `8DEEDC0338273B81EAD6BC89D8C12D53E6091906390EC81C5EC9CE37170CB3EC` | `8DEEDC0338273B81EAD6BC89D8C12D53E6091906390EC81C5EC9CE37170CB3EC` | **100% Match** |
| `outputs/combined_splits.csv` | `07F94433D6BB638EDE552EF9338F0E2258BEAFBEFB85F487CDC58A6DD7548149` | `07F94433D6BB638EDE552EF9338F0E2258BEAFBEFB85F487CDC58A6DD7548149` | **100% Match** |
| `config.json` | `AD23CC09A5050C5B390AF56BCE25B389CAC72E1BB52E3416F8A48C08801DEDB1` | `AD23CC09A5050C5B390AF56BCE25B389CAC72E1BB52E3416F8A48C08801DEDB1` | **100% Match** |

**Conclusion:** Zero drift, zero corruption, zero modifications across all protected checkpoints.

---

## 3. Dataset Audit: Types, Labels, and Scientific Scope

### A. Cosmetic Skin-Type Dataset (`archive (2)\Oily-Dry-Skin-Types`)
* **Path:** `D:\DOWNLOADS D\archive (2)\Oily-Dry-Skin-Types`
* **Volume:** 3,152 images
* **Image Modality:** Facial macro-portraits (forehead, cheeks, nose, chin).
* **Labels:** `dry` (758), `normal` (1,274), `oily` (1,120).
* **Critical Finding & Label Meaning:**
  In cosmetic science, "normal" skin refers to **eudermic skin** (an optimal sebum/moisture equilibrium in healthy facial skin). It is used to select moisturizers versus oil-control cosmetics. 
  **It has zero diagnostic validity for skin pathology.** A face labeled "normal" in this dataset can still have melanoma, basal cell carcinoma, psoriasis, or fungal infections; conversely, "oily" skin is a physiological characteristic, not a disease.
  **Decision:** Excluded from medical disease evaluation and cannot be used as a "normal-vs-disease" ground truth.

### B. 11k Hands Dataset (`data/experiments/normal_hand_v1/manifest_v1.csv` / `archive (3)`)
* **Volume:** 11,076 images across 189 individuals.
* **Modality:** Macroscopic dorsal and palmar photographs taken under studio lighting on uniform white matte paper.
* **Subjects:** Healthy volunteers without documented dermatological conditions. Four individuals (243 images) exhibited visible surface irregularities/atypical marks and were quarantined as an exploratory evaluation set (Subjects 27, 570, 1566, 1000000).
* **Scope:** Suitable strictly for modeling healthy hand appearance density on white backgrounds.

### C. Clinical Hand & Body Datasets (`archive (4)`)
* **Volume:** 2,147 clinical images (`eczema`: 1,122; `tinea`: 1,025).
* **Audit Findings:** MediaPipe hand detection shows that **only 13.3% of eczema** images and **3.3% of tinea** images actually contain hands. The vast majority depict legs, backs, torsos, and feet on hospital sheets.
* **Shortcut Warning:** Confounding by background: training a model on 11k Hands vs `archive (4)` would yield an artificial classifier that detects "white studio paper vs hospital bedding", not skin health.

### D. Webcam Hand Normal Cohort (`data_clinical\normal_webcam`)
* **Path:** `data_clinical\normal_webcam`
* **Current State:** Directory created with manifest header `people.csv` (`person_id,age_group,skin_tone_self_report,device,lighting,hand_side,consent_date`).
* **Images:** Awaiting consented user uploads.

---

## 4. Trained Artifacts Audit & Model-to-Task Mapping

| Model Path | Architecture | Input Size | Training Dataset | Actual Trained Task | Can It Act as Normal-vs-Abnormal? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `models/best_model.keras` | EfficientNetB0 | $224 \times 224 \times 3$ | HAM10000 (Dermoscopy) | 7-class lesion classifier (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) | **NO** (Has no "normal" class; dermoscopy optics only) |
| `models/isic_augmented_v1.keras` | EfficientNetB0 | $224 \times 224 \times 3$ | HAM10000 + Augmentation | 7-class lesion classifier | **NO** (Dermoscopy optics only) |
| `models/isic_finetuned.keras` | EfficientNetB0 | $224 \times 224 \times 3$ | HAM10000 + ISIC 2019 | 7-class lesion classifier | **NO** (Dermoscopy optics only) |
| `models/ood_gate.npz` | MobileNetV3 + Mahalanobis | $224 \times 224 \times 3$ | HAM10000 Features | Dermoscopy Out-of-Distribution filter | **NO** (Only flags deviation from dermoscopy) |
| `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` | MobileNetV3Small + PCA-Mahalanobis ($k=48$) | $224 \times 224 \times 3$ (Hand Crop) | 11k Hands (129 healthy individuals) | One-class hand appearance density estimation | **NO** (Fitted only on normal hands; never evaluated on clinical lesions) |
| `models/normal_disease/` | — | — | — | *None (Empty folder)* | **NO (No artifact exists)** |
| `models/normal_vs_condition/` | — | — | — | *None (Empty folder)* | **NO (No artifact exists)** |

### Explicit Technical Finding regarding `hand_anomaly_baseline.npz` (Per Addition 3):
The Stage D/G hand detector is an **isolated one-class appearance detector** fitted exclusively on studio-background 11k Hands images. It measures statistical distance from a healthy dorsal/palmar studio baseline. **It was never evaluated on abnormal hands or clinical hand pathologies.** It has no sensitivity data for hand dermatitis, melanoma, or fungal infections. **It cannot mathematically or clinically be described or deployed as a normal-vs-abnormal classifier.**

---

## 5. Architectural Design for Stage H Integration

To address the user's workflow without violating scientific integrity or fabricating non-existent classifiers, the inference pipeline will be structured into three distinct stages:

```
[Raw Image Input (Upload or Live Webcam)]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Input Validation & Image Suitability Gate          │
│ • Image decodability & non-corruption check                 │
│ • Minimum resolution (128x128) & brightness/blur check      │
│ • Human Skin Suitability: HSV / YCrCb skin chrominance      │
│   mask area (> 15% minimum skin coverage)                   │
│ • Rejection of non-skin: Cars, anime, gaming posters,       │
│   unrelated objects, pure noise                             │
│ • Display on failure: "No valid skin image detected."       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Passed (Valid Skin Present)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Modality & Anatomical Routing                      │
│ • Hand Detector (MediaPipe Hand Landmark Localization)      │
│   ├── If Hand Detected:                                     │
│   │   Show: "Hand detected. This system cannot yet tell     │
│   │          normal skin from a skin condition."            │
│   │   (No condition label, no 7-class score table shown)    │
│   │                                                         │
│   └── If Hand NOT Detected (Close-up lesion image):         │
│       Proceed to Stage 3 Dermoscopy Gate                    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Lesion Close-up Path
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Existing 7-Class Dermoscopic Classifier            │
│ • Out-of-Distribution (OOD) Gate (src/gate.py)              │
│   ├── If OOD Gate Rejects:                                  │
│   │   Show: "No condition label shown. This photo is       │
│   │          outside the validated domain (dermoscopy)."    │
│   │                                                         │
│   └── If OOD Gate Accepts (Validated Dermoscopy Domain):    │
│       Execute EfficientNetB0 7-class ensemble with          │
│       calibrated decision threshold (t = 0.347)             │
│       Display prediction, confidence, and melanoma flag     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Fixtures & Test Plan (Per Addition 5)

### Existing Fixtures in `tests\fixtures\`:
1. `gaming_poster.jpg` (Anime / non-skin artwork)
2. `webcam_classroom_hand.jpg` (Hand in non-dermoscopic classroom environment)

### Required Test Suite Categories (Per-Case Outcome Matrix):
To be validated upon receiving real consented test fixtures:
1. `normal_webcam_hand.jpg` (Real hand photo) $\rightarrow$ Expected: **Stage 2 Hand Banner ("cannot yet tell normal skin from a skin condition")**, 7-class classifier blocked.
2. `hand_with_accessories.jpg` (Rings/watch) $\rightarrow$ Expected: **Stage 2 Hand Banner**, 7-class classifier blocked.
3. `oily_skin_photo.jpg` (Facial close-up) $\rightarrow$ Expected: Rejected by OOD / outside validated dermoscopy domain.
4. `hand_with_condition.jpg` (Real lesion on hand) $\rightarrow$ Expected: **Stage 2 Hand Banner**, 7-class classifier blocked.
5. `dermoscopy_closeup.jpg` (HAM10000 validation split) $\rightarrow$ Expected: **Passed through Stages 1, 2, and 3** to produce 7-class prediction.
6. `gaming_poster.jpg` / `car_or_anime.jpg` $\rightarrow$ Expected: **Rejected at Stage 1 ("No valid skin image detected.")**.
7. `corrupted_file.bin` $\rightarrow$ Expected: **Rejected at Stage 1 ("No valid skin image detected.")**.
8. `blurred_photo.jpg` $\rightarrow$ Expected: **Rejected at Stage 1 (Low quality / blur check)**.

---

## 7. What Would Close the Normal-vs-Abnormal Hand Gap?

To train and validate a true binary normal-vs-abnormal hand model without confounding shortcuts:
1. **Consented Webcam Normal Hands (`data_clinical/normal_webcam`):** At least 50–100 individuals across diverse Fitzpatrick skin tones (I–VI), ages, and real-world lighting conditions (ambient daylight, indoor LED, shadows) with metadata in `people.csv`.
2. **Clinically Verified Hand Lesions:** At least 200–500 confirmed clinical photographs of common hand pathologies (e.g., hand dermatitis/eczema, tinea manuum, psoriasis, verruca vulgaris, acral lentiginous melanoma) captured under similar consumer phone/webcam lighting.
3. **Person-Disjoint Splits:** Stratified 70/15/15 splits by subject ID to prevent identity leakage.

---

## 8. Checkpoint & Approval Gate

- [x] Repository audit complete.
- [x] Protected assets verified 100% bit-for-bit against locked zip.
- [x] Datasets and label schemes documented.
- [x] Scientific limitation of one-class hand baseline explicitly established.
- [x] Architecture plan formulated.
- [x] Feature flag `ENABLE_NORMAL_HAND_DETECTOR` remains `False`.
- [x] Zero application files modified.

**STOP.** Per Addition 4, stopping here for user written approval before writing any integration code.
