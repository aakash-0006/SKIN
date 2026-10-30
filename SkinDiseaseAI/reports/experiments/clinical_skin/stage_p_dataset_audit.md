# Stage P — Clinical Skin Dataset Audit and Model Development Preparation Report

**Date:** 2026-10-30  
**Author:** Senior Computer Vision & Deep-Learning Engineer  
**Project:** SkinDiseaseAI — Clinical Smartphone Image Analysis Pipeline  
**Intended Output Categories:** Normal-looking skin appearance, Potentially abnormal skin appearance, Uncertain/unsupported image  
**Safety Classification:** Educational research prototype — NOT a medical diagnostic device  
**Report Destination:** `reports/experiments/clinical_skin/stage_p_dataset_audit.md`  

---

## 1. Executive Summary

As SkinDiseaseAI expands from dermoscopic lesion analysis (HAM10000, ISIC 2019, ISIC 2024) toward everyday smartphone photographs, we conducted an exhaustive forensic audit across all clinical datasets present on the system.

### Key Forensic Audit Findings:
1. **Stanford Diverse Dermatology Images (DDI):**
   - **Status:** **Fully Downloaded, Extracted, and Audited.**
   - **Location:** `D:\DOWNLOADS D\ddidiversedermatologyimages\`
   - **Content:** Exactly 656 photographic clinical images with complete metadata (`ddi_metadata.csv`), spanning 78 disease conditions and balanced across Fitzpatrick skin types (FST I–VI).
   - **Label Reality:** Contains **zero** normal healthy skin controls. It consists of 485 benign lesions and 171 biopsy-confirmed malignant neoplasms.
2. **Fitzpatrick17k:**
   - **Status:** **Metadata and Code Repositories Extracted; Raw Images NOT Downloaded.**
   - **Location:** `D:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI\data_normal_disease\fitzpatrick17k_repo\`
   - **Content:** Contains complete metadata for 16,577 records (`fitzpatrick17k.csv`) referencing external web URLs from DermaAmin and AtlasDerm. Raw image files do not reside on local disk.
   - **Label Reality:** Contains 114 skin disease labels (acne, eczema, psoriasis, basal cell carcinoma, melanoma). There are **zero** normal healthy skin images.
3. **SCIN (Skin Condition Image Network):**
   - **Status:** **Schema, Documentation, and Repository Extracted; Cloud Bucket NOT Downloaded.**
   - **Location:** `D:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI\data_normal_disease\scin_repo\`
   - **Content:** Open-access research framework from Google Research / Stanford Medicine. Data resides in the Google Cloud Storage bucket (`gs://dx-scin-public-data`). 5,749 crowdsourced contributions from internet search users.
   - **Label Reality:** 89% inflammatory, infectious, or allergic conditions (eczema, contact dermatitis). Contributor queries were dermatological complaints; normal healthy skin was not collected.
4. **Existing Normal-Hand Dataset (11k Hands):**
   - **Status:** **Fully Downloaded, Extracted, Audited, and Trained into Candidate 1B.**
   - **Location:** `C:\Users\USER\Downloads\normal skin dataset\archive (3)\Hands\Hands\`
   - **Content:** Exactly 11,076 uncompressed high-resolution images of healthy human hands across 190 subjects. 10,912 images without irregularities.
   - **Detector Status:** `Candidate 1B` (L2-normalized MobileNetV3 + 32-dim PCA + Mahalanobis distance) is verified in `src/normal_hand/pipeline.py` and passes 9/9 unit tests.
5. **Cosmetic Facial Datasets:**
   - **Location:** `C:\Users\USER\Downloads\normal skin dataset\archive (2)\Oily-Dry-Skin-Types`
   - **Finding:** Cosmetic classification (dry vs. oily vs. normal cosmetic sebum levels). In accordance with strict rules, cosmetic skin types **cannot** be used as medical disease or normal controls.

---

## 2. Task 1: Detailed Dataset Discovery and Forensic Inventory

| Dataset | Root Path on Machine | Total Images | Annotation Files | Image Format & Dimensions | Skin-Tone Metadata | Patient Identifiers | Licensing & Permitted Usage | Lifecycle Status |
|:---|:---|:---:|:---|:---|:---:|:---:|:---|:---:|
| **Stanford DDI** | `D:\DOWNLOADS D\ddidiversedermatologyimages\` | **656** | `ddi_metadata.csv` (656 rows) | PNG (155×79 to 1914×1429, mean 900×700) | `skin_tone`: 12 (208), 34 (241), 56 (207) | `DDI_ID` (1 to 656) — Lesion level, no multi-lesion patient IDs | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) | **Extracted & Audited** |
| **Fitzpatrick17k** | `data_normal_disease/fitzpatrick17k_repo/` | **0 local** (16,577 URLs) | `fitzpatrick17k.csv` (16,577 rows) | External URLs (DermaAmin, AtlasDerm); uncompressed JPEGs | `fitzpatrick_scale` (1 to 6) & `fitzpatrick_centaur` | `url_alphanum` / `md5hash` — No patient-level ID | Academic Research Only (Fair Use Educational Derivative) | **Metadata Extracted; Images Not Downloaded** |
| **Google SCIN** | `data_normal_disease/scin_repo/` | **0 local** (5,749 cloud cases) | `dataset_schema.md`, `scin_demo.ipynb` | Google Cloud Storage bucket (`gs://dx-scin-public-data`) | Estimated FST (eFST), Self-reported FST (sFST), Monk Scale (eMST) | Case ID (`case_id`) — Crowdsourced anonymous contributions | SCIN Data Use License (Educational / Non-Commercial Research) | **Repo Extracted; Cloud Bucket Not Downloaded** |
| **11k Normal Hands** | `C:\Users\USER\Downloads\normal skin dataset\archive (3)\Hands\Hands\` | **11,076** | `HandInfo.csv` (11,076 rows) | JPEG ($1600 \times 1200$, uniform RGB) | `skinColor` (fair, medium, dark self-reported) | `id` (190 distinct subjects, 11,076 images) | Research and Educational Use | **Trained & Validated (Candidate 1B)** |
| **Eczema vs Tinea** | `C:\Users\USER\Downloads\normal skin dataset\archive (4)\` | **1,000+** | Directory hierarchy (`train/`, `test/`) | JPEG (variable web resolution) | None | None (web scraped clinical cases) | Non-commercial research | **Extracted; Needs Quality Filtering** |

---

## 3. Task 2: Label Verification & Semantic Categorization

To avoid misleading clinical models, labels must be strictly partitioned into genuine clinical categories:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          CLINICAL SKIN IMAGE TAXONOMY                                  │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ 1. Normal-Looking Skin   │ 2. Visible Skin Condition   │ 3. Unsupported / Uncertain    │
│    (No Visible Lesion)   │    (Lesion, Rash, Tumor)    │    (Rejection Gate)           │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ • 11k Normal Hands       │ • Malignant: Melanoma, BCC, │ • Non-Skin (Cars, Anime, Art) │
│   (irregularities = 0)   │   SCC, Mycosis fungoides    │ • Corrupted / Low Illum / Blur│
│ • Clear webcam skin      │ • Benign: Seborrheic kerat- │ • Unsegmented cluttered image │
│   (patient-consented)    │   osis, Nevus, Verruca      │ • Borderline / Ungradable     │
│ • Derm-verified clear    │ • Inflammatory: Eczema,     │                               │
│   skin tiles             │   Psoriasis, Tinea, Acne    │                               │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Critical Label Integrity Guardrails:
1. **Benign Lesions Are NOT Normal Skin:**
   - In DDI, 485 lesions are benign (seborrheic keratosis, melanocytic nevi, dermatofibroma, acrochordon).
   - A benign keratosis or giant mole is **visible dermatological pathology**, not clear healthy skin. Feeding benign lesions as negative controls would teach the model that moles and keratoses represent normal healthy skin.
2. **Cosmetic Types Are NOT Medical Labels:**
   - Cosmetic sebum classifications (dry vs. oily) in `archive (2)` have no clinical ICD-10 or SNOMED pathology. They are excluded from the medical classifier.
3. **Missing Annotations:**
   - In SCIN and Fitzpatrick17k, records with missing disease annotations are ungradable or multi-condition cases; they are **never** treated as normal skin.

---

## 4. Task 3: Dataset Compatibility & Leakage Prevention

### 4.1 Image Domain & Acquisition Discrepancies
- **Stanford DDI:** Macro clinical photography of focal skin lesions taken in clinical examination rooms under medical lighting. High lesion-to-background ratio.
- **Fitzpatrick17k:** Scraped from clinical dermatology atlases. Often contains surgical margins, rulers, purple ink markers, and variable lighting.
- **SCIN:** Real-world smartphone photographs taken by patient-users at home. Highly variable distance, flash reflections, and background clutter (clothing, bedsheets, flooring).
- **11k Hands:** Controlled studio setup with a uniform white matte background and fixed camera distance.

### 4.2 Anatomical Site Breakdown
- **Stanford DDI:**
  - Torso, extremities, head/neck, and acral hand/foot lesions. Acral lentiginous melanoma samples specifically focus on palms, soles, and fingers.
- **11k Hands:**
  - 100% human hands (dorsal right, dorsal left, palmar right, palmar left).
- **SCIN:**
  - Fully annotated with body location tags (`body_site` including face, hands, trunk, legs).

### 4.3 Patient-Level Leakage Protection
- In DDI, each image is a distinct lesion (`ddi_1` to `ddi_656`). There are no patient ID groupings provided; cross-validation splits must use stratified random grouping.
- In 11k Hands, subject IDs (`id` 1 to 190) **must** be grouped so that images from the same subject never appear in both train and validation.

---

## 5. Task 4: Proposed Clinical Skin Model Architecture

To handle everyday phone photographs without compromising clinical safety, the pipeline follows a multi-stage hierarchical routing architecture:

```
[Uploaded Smartphone Image]
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Input Quality & Skin Localization Gate             │
│ • Blur Detection (Laplacian Variance >= 100)               │
│ • Illumination Check (15 <= Mean Pixel Intensity <= 245)    │
│ • Color Contrast & Resolution Check (>= 128x128)            │
│ • OOD Non-Skin Gate (Mahalanobis feature distance)          │
└─────────────────────────────────────────────────────────────┘
             │ Passed Quality & Non-Skin Rejection
             ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Anatomical Routing & Normal-Hand Gate              │
│ • MediaPipe Hand Landmark Detector                          │
│   ├── Hand Detected: Route to Candidate 1B Normal-Hand      │
│   │   ├── Distance <= 6.931 ──> [Normal-Looking Hand Skin]  │
│   │   └── Distance > 6.931  ──> [Route to Clinical Engine]  │
│   └── Non-Hand Skin: Route to General Clinical Engine       │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Clinical Normal vs. Abnormal Appearance Engine     │
│ • Architecture: EfficientNetB0 or MobileNetV3Large          │
│ • Output 1: P(Normal-looking skin appearance)               │
│ • Output 2: P(Potentially abnormal condition appearance)    │
│ • Output 3: P(Uncertain / Needs clinical in-person review)  │
│ • Confidence Thresholding: Entropy & Softmax Margin Gate    │
└─────────────────────────────────────────────────────────────┘
```

### Clear Operational Separation:
- **Production 7-Class Dermoscopic Classifier (`best_model.keras`):** Remains strictly isolated for close-up dermoscopy inputs.
- **ISIC 2024 Binary Malignancy Model (`isic2024_fusion_fold0_best.keras`):** Remains strictly isolated for 3D TBP tile and clinical metadata inputs.
- **Clinical Smartphone Model:** Handles natural-light everyday camera captures, providing clear screening categorization without medical diagnostic claims.

---

## 6. Task 5: Strict Preservation Compliance & Checksums

In accordance with mandatory safety rules:
- **Production Checkpoints Untouched:** `models/best_model.keras`, `models/isic_finetuned.keras`, `models/isic_augmented_v1.keras`, and `models/ood_gate.npz` remain completely unmodified.
- **Production Pipeline Untouched:** `app.py`, `src/predict.py`, and `src/hand_validator.py` remain completely unmodified.
- **ISIC 2024 Research Untouched:** `outputs/isic2024/stage_o2/` and `models/isic2024_stage_o2/` remain completely intact.
- **Candidate 1B Checkpoint Preserved:** `models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz` remains untouched.
- **Zero Training Executed:** No training was launched; this stage was exclusively an audit and design preparation.

---

## 7. Dataset Lifecycle Matrix & Next Steps

| Dataset | Downloaded | Extracted | Audited | Preprocessed | Trained | Validated | Integrated |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stanford DDI (656 images)** | Yes | Yes | **Yes (Stage P)** | Ready | No | No | No |
| **Fitzpatrick17k (16.5k rows)** | Metadata Only | Yes | **Yes (Stage P)** | Pending Download | No | No | No |
| **Google SCIN (5.7k cases)** | Repo Only | Yes | **Yes (Stage P)** | Cloud Bucket | No | No | No |
| **11k Normal Hands** | Yes | Yes | **Yes (Stage P)** | Yes | **Yes (Candidate 1B)** | **Yes (9/9 tests)** | Toggled (`False`) |
| **Eczema vs Tinea (1k images)** | Yes | Yes | **Yes (Stage P)** | Needs QC | No | No | No |

### Recommended Action Plan for User Approval:
1. **Immediate Use of Stanford DDI:** DDI is 100% downloaded, uncompressed, and audited on disk (`D:\DOWNLOADS D\ddidiversedermatologyimages`). It can immediately serve as a verified benchmark for clinical condition classification across skin tones FST I–VI.
2. **Combine with Normal Hand Controls:** The 11,076 normal hand images provide the first verified, leak-free source of genuine normal-looking skin controls.
3. **Stop & Await Approval:** Standing by for your review of this audit report before downloading external images or building training scripts.
