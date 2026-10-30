# Stage M — ISIC 2024 Dataset Integration Audit and Training Preparation Report

**Date:** 2026-10-30  
**Author:** Senior Machine Learning Engineer, Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI  
**Audited Location:** `D:\DOWNLOADS D\isic-2024-challenge`  
**Existing Production Manifest:** `outputs/combined_splits.csv` (16,546 train / 3,440 val / 1,494 test)  
**Existing Production Models:** 7-Class Dermoscopic Classifiers (`best_model.keras`, `isic_finetuned.keras`, `isic_augmented_v1.keras`)  
**Safety Classification:** Educational research prototype — NOT a medical device  

---

## 1. Executive Summary & Compatibility Decision

### **COMPATIBILITY DECISION: INCOMPATIBLE WITH SEVEN-CLASS DERMOSCOPIC MANIFEST**
### **MANDATORY RECOMMENDATION: SEPARATE BINARY MALIGNANCY RESEARCH PIPELINE**

A rigorous, programmatic Phase 1 inspection and Phase 2 compatibility analysis of the **ISIC 2024 Challenge Dataset** (`D:\DOWNLOADS D\isic-2024-challenge`) was conducted.

### Core Audit Conclusions:
1. **Severe Label Incompatibility (Binary vs 7-Class):**  
   The ISIC 2024 challenge is structured as a **binary lesion malignancy detection task** (`target`: 0 = benign, 1 = malignant). Exactly **99.73% of the dataset (399,991 of 401,059 images) has NO 7-class disease label** and is annotated solely as `Benign`. Forcing this binary target into the existing 7-class taxonomy (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) is **scientifically invalid and clinically unsafe**.
2. **Severe Image Domain Shift (3D TBP Crops vs Optical Dermoscopy):**  
   100% of the 401,059 images in ISIC 2024 are **3D Total-Body Photography (TBP) tile crops** ($139 \times 139$ square tiles, median file size $2.8\text{ KB}$) captured from Vectra WB360 whole-body multi-camera rigs. They are **not optical dermoscopy close-ups**. When passed through the existing production Out-of-Distribution gate (`models/ood_gate.npz`), **36.0% are immediately rejected as out-of-domain** due to blur, low optical resolution, and lack of dermoscopic structures.
3. **Extreme Class Imbalance:**  
   Out of 401,059 images, only **393 are malignant** (**0.0980%**, or 1 positive per 1,020 images). 
4. **Data Isolation Action Taken:**  
   In strict accordance with Phase 2 preservation rules:
   - ISIC 2024 **WAS NOT ADDED** to `outputs/combined_splits.csv`.
   - The existing 7-class training manifest (21,480 total rows across HAM10000 and ISIC 2019) remains **100% untouched and protected**.
   - No models were retrained or overwritten.
   - A separate, dedicated binary malignancy pipeline has been designed for review.

---

## 2. Phase 1 — Technical Audit of the ISIC 2024 Dataset

### 2.1 File Inventory

| File / Folder Name | File Type | Record Count / Dimensions | File Size | Description & Status |
|:---|:---|:---:|:---:|:---|
| **`train-metadata.csv`** | CSV Table | 401,059 rows, 55 columns | **245.6 MB** | Ground-truth labels, patient IDs, metadata, and 3D TBP engineered features |
| **`train-image.hdf5`** | HDF5 Archive | 401,059 JPEG image arrays | **1,237.2 MB** | Serialized $139 \times 139$ square JPEG tiles indexed by `isic_id` |
| **`test-metadata.csv`** | CSV Table | 3 rows, 46 columns | **2.2 KB** | Mock test metadata for Kaggle code submission validation |
| **`test-image.hdf5`** | HDF5 Archive | 3 JPEG image arrays | **10.4 KB** | Mock test images for Kaggle code submission validation |
| **`sample_submission.csv`** | CSV Table | 3 rows, 2 columns | **66 bytes** | Template format: `isic_id`, `target` |

> [!IMPORTANT]
> The unlabelled test images in `test-image.hdf5` consist of only 3 mock entries designed for competition runtime checking; actual test evaluation in ISIC 2024 is held in private rerun environments. Consequently, **100% of local experimentation must rely strictly on cross-validation within `train-metadata.csv` and `train-image.hdf5`**.

---

### 2.2 Patient, Lesion, and Image Identifiers

- **Total Images (`isic_id`):** `401,059` unique images.
- **Total Patients (`patient_id`):** `1,042` unique individuals.
- **Images per Patient:** Mean = **`384.9`**, Median = **`352`**, Min = **`1`**, Max = **`9,184`**.
- **Identified Lesions (`lesion_id`):** `22,058` annotated lesion tracking IDs; `379,001` records have missing/untracked lesion IDs.
- **Data Leakage Risk:** Because each patient contributes hundreds of lesion tiles across their body, **any random image-level train/validation split will cause catastrophic patient leakage**. Splitting **MUST** be performed strictly at the `patient_id` group level.

---

### 2.3 Image Modality & Optical Properties

- **Image Type (`image_type`):** 100% `TBP tile: close-up`.
- **Acquisition Rig (`tbp_tile_type`):**
  - `3D: XP` (cross-polarized 3D TBP): 285,903 images (71.3%).
  - `3D: white` (white light 3D TBP): 115,156 images (28.7%).
- **Image Resolution:** Uniform $1.0$ aspect ratio (square crops); image dimensions range tightly around **$117 \times 117$ to $145 \times 145$ pixels** (mean: $135 \times 135$).
- **File Compression:** Stored as compressed JPEG byte-streams inside HDF5; median file size per image is **`2,844 bytes`**.
- **Anatomical Site Distribution (`anatom_site_general`):**
  - Posterior torso: `121,902` (30.4%)
  - Lower extremity: `103,028` (25.7%)
  - Anterior torso: `87,770` (21.9%)
  - Upper extremity: `70,557` (17.6%)
  - Head / neck: `12,046` (3.0%)
  - Unrecorded: `5,756` (1.4%)

---

### 2.4 Label Semantics & Diagnostic Breakdown

The dataset contains a primary binary classification target:
- **`target = 0` (Benign):** `400,666` images (**`99.902%`**)
- **`target = 1` (Malignant / Biopsy-Confirmed Skin Cancer):** `393` images (**`0.0980%`**)

#### Fine-Grained Diagnosis Audit (`iddx_full`):
- **`Benign` (generic, no biopsy):** **`399,991` images (99.73%)**
- **Specific Biopsy-Confirmed Lesions:** Only **`1,068` images (0.27%)** have histological or dermatological sub-diagnoses:

| Diagnostic Subcategory (`iddx_full`) | Clinical Superclass | Target | Image Count | Existing 7-Class Alignment |
|:---|:---|:---:|:---:|:---|
| **Basal cell carcinoma (BCC)** (Nodular, Superficial, Infiltrating) | Malignant | 1 | **163** | Direct match to `bcc` |
| **Melanoma** (In situ, Invasive, Lentigo maligna, Nodular) | Malignant | 1 | **157** | Direct match to `mel` |
| **Squamous cell carcinoma (SCC / SCC in situ)** | Malignant | 1 | **73** | Corresponds partially to `akiec` |
| **Nevus** (Atypical, Dysplastic, Clark, Compound, Dermal) | Benign | 0 | **443** | Direct match to `nv` |
| **Atypical melanocytic neoplasm** | Indeterminate | 0 | **64** | Borderline / Indeterminate |
| **Seborrheic keratosis** | Benign | 0 | **56** | Match to `bkl` |
| **Actinic / Solar keratosis** | Indeterminate | 0 | **38** | Match to `akiec` |
| **Fibro-histiocytic / Dermatofibroma** | Benign | 0 | **15** | Match to `df` |
| **Vascular lesions** | Benign | 0 | **0** | **0% coverage** (`vasc` missing) |
| **Generic Benign (Untyped)** | Benign | 0 | **399,991** | **CANNOT MAP (99.73% of data)** |

> [!CAUTION]
> **Why `target = 0` CANNOT be treated as "Normal Healthy Skin":**
> In ISIC 2024, `target = 0` denotes a non-malignant skin lesion (e.g. dysplastic nevi, seborrheic keratoses, solar lentigines, angiomas, scars). It does **NOT** denote normal, lesion-free skin. Treating `target = 0` as normal skin would teach the model that clinical nevi and keratoses represent clear, healthy skin.

---

### 2.5 Dataset Provenance & Copyright Licensing

- **Contributing Institutions (`attribution`):**
  - Memorial Sloan Kettering Cancer Center (MSKCC): `129,068` (32.2%)
  - Hospital Clínic de Barcelona: `105,724` (26.4%)
  - University Hospital of Basel: `65,218` (16.3%)
  - The University of Queensland (Frazer Institute): `51,768` (12.9%)
  - ACEMID MIA (Australia): `28,665` (7.1%)
  - Medical University of Vienna (ViDIR): `12,640` (3.2%)
  - Andreas Syggros Hospital (Athens): `7,976` (2.0%)
- **Licensing Breakdown (`copyright_license`):**
  - `CC-BY`: `188,812` (47.1%)
  - `CC-BY-NC` (Non-Commercial Only): `183,582` (45.8%)
  - `CC-0` (Public Domain): `28,665` (7.1%)
- **Legal Restriction:** Because **45.8%** of the dataset is governed by `CC-BY-NC`, any model trained on this combined dataset is legally restricted to **non-commercial academic and educational research**.

---

## 3. Phase 2 — Multi-Dimensional Compatibility Assessment

| Domain Dimension | Existing Seven-Class Dataset (HAM10000 + ISIC 2019) | ISIC 2024 Challenge Dataset | Compatibility Finding |
|:---|:---|:---|:---|
| **Clinical Objective** | 7-Class lesion differential diagnosis (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) | Binary malignancy triage (`target`: 0 = benign, 1 = malignant) | **INCOMPATIBLE** |
| **Label Completeness** | 100% of images mapped to confirmed 7 classes | Only **0.27%** have disease labels; **99.73%** are generic `Benign` | **FATAL LABEL GAP** |
| **Image Modality** | Contact & non-contact **optical dermoscopy** | **3D Total-Body Photography (TBP) crops** from Vectra WB360 | **FATAL DOMAIN MISMATCH** |
| **Optical Resolution** | High optical detail ($600 \times 450$ up to $1024 \times 1024$) | Digital crops ($117 \times 117$ to $145 \times 145$, mean $135 \times 135$) | **SEVERE DISPARITY** |
| **Dermoscopic Structures** | Pigment networks, blue-white veils, globule patterns, arborizing vessels clearly visible | Digital tile crops lack microscopic sub-surface optical structures | **OPTICAL MISMATCH** |
| **OOD Gate Acceptance** | 100% In-Distribution training reference | **36.0% rejected** by existing `ood_gate.npz` | **FAILS PRODUCTION GATE** |
| **Positive Rate** | Balanced / Moderately imbalanced (Melanoma ~11%) | Extreme imbalance: **0.0980%** malignant (1 in 1,020) | **EXTREME IMBALANCE** |
| **Data Partitioning** | Lesion-disjoint GroupShuffleSplit (HAM) | Patient-disjoint GroupKFold on `patient_id` mandatory | Separate partition structure required |

### Compatibility Verdict:
**ISIC 2024 CANNOT BE ADDED TO `outputs/combined_splits.csv`.**  
Adding 400,000 generic `Benign` images would dilute the 7-class disease distribution by a factor of 20, destroying the learned representation of `best_model.keras` and triggering massive false rejections at the OOD gate.

---

## 4. Phase 3 & 4 — Recommended Strategy: Dedicated Binary Malignancy Model

Rather than corrupting the 7-class dermoscopic pipeline, ISIC 2024 represents a powerful asset for a **secondary, independent binary cancer screening model** designed specifically for total-body photography and teledermatology triage.

### 4.1 Architecture & Pipeline Design

```
[3D TBP Image Tile (139x139)] 
               │
               ▼
┌─────────────────────────────────┐
│ EfficientNet-B0 / MobileNetV3   │ ──► [GlobalAvgPool (1280-dim)] ──┐
│ (Backbone Image Encoder)        │                                  │
└─────────────────────────────────┘                                  │
                                                                     ▼
[Tabular Metadata: 45 features]   ──► [MLP / Dense Layers (64-dim)] ──► [Concat] ──► [Dense(1, sigmoid)] ──► Malignancy Probability
(tbp_lv_*, age, sex, site)
```

1. **Multimodal Fusion:**  
   ISIC 2024 is uniquely suited for multimodal learning because it includes 45 engineered lesion features (`tbp_lv_areaMM2`, `tbp_lv_color_std_mean`, `tbp_lv_deltaLBnorm`, `tbp_lv_nevi_confidence`, `tbp_lv_eccentricity`, `clin_size_long_diam_mm`). Combining image embeddings with tabular metadata yields state-of-the-art triage accuracy.
2. **Patient-Stratified Split:**  
   - Use `StratifiedGroupKFold(n_splits=5)` grouped strictly by `patient_id` and stratified by `target`.
   - Guaranteed zero patient overlap between train and validation.
3. **Class Imbalance Strategy:**  
   - With 393 positives in 401,059 images, training on all 400,000 negatives per epoch causes gradient vanishing.
   - **Recommended Training Batching:** Dynamic negative sampling (e.g. 1 positive : 15 negatives per mini-batch) or focal loss ($\gamma = 2.0$, $\alpha = 0.25$).
   - **Validation Integrity:** Validation partitions must preserve the natural un-resampled prevalence ($0.098\%$) to reflect true clinical performance.
4. **Target Evaluation Metric:**  
   - Standard accuracy is meaningless ($99.9\%$ accuracy is achieved by predicting all zeros).
   - Primary metric: **Partial Area Under ROC Curve ($p\text{AUC}$) above 80% True Positive Rate** (the clinical metric of the ISIC 2024 challenge), combined with Precision-Recall AUC (PR-AUC), Sensitivity at 80% Specificity, and Specificity at 80% Sensitivity.

---

## 5. Preservation & Integrity Sign-Off

- **Production Asset Integrity:**  
  `outputs/combined_splits.csv`, `outputs/splits.csv`, `models/best_model.keras`, `models/isic_finetuned.keras`, `models/isic_augmented_v1.keras`, `models/ood_gate.npz`, and `config.json` remain **100% bit-for-bit identical** to their locked backup states.
- **Normal-Hand Integrity:**  
  Candidate 1B normal-hand detector (`candidate_1b_l2_mahalanobis_k32.npz`) and baseline detector remain unmodified.
- **Feature Flag Status:**  
  `ENABLE_NORMAL_HAND_DETECTOR = False` preserved.
- **Ordinary Application:**  
  `app.py`, camera hardware controls, and UI remain unmodified.

---

## 6. Stop Condition & Next Steps

In strict compliance with instructions:
- Dataset audit and compatibility analysis are complete.
- The existing manifest has **NOT** been modified.
- No training has been initiated.
- **STOPPED and awaiting your review and approval** to implement the separate binary malignancy pipeline (`src/isic2024_binary/`) or proceed with your preferred strategy.
