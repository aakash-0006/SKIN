# 📊 SkinDiseaseAI — Stage A: Three-Dataset Physical Audit & Compatibility Report

**Project:** SkinDiseaseAI — Full Dataset Integration, Augmentation, Retraining and Accuracy Improvement  
**Author:** Senior Machine Learning Engineer, Deep Learning Researcher & Medical CV Specialist  
**Date:** October 29, 2026  
**Document:** `reports/experiments/three_dataset_audit.md`  
**Execution Environment:** Windows 11 | Python 3.12.10 | TensorFlow 2.21.0 | 24 GB RAM | NVIDIA RTX 3050  
**Status:** Stage A Read-Only Dataset Audit Complete — Stopped for User Review  

---

## 1. Executive Summary & Audit Scope

In accordance with **Stage A** of the execution protocol, this report presents an exhaustive, physical on-disk audit of the three candidate dermatological datasets available on the host machine:
1. **HAM10000** (Dermoscopy baseline)
2. **ISIC 2019** (Dermoscopy expansion)
3. **Stanford DDI** (Diverse Dermatology Images — Clinical photography external benchmark)

### Core Findings at a Glance
- **All Three Datasets Physically Exist on Disk:** Exactly 10,015 HAM10000 images, 25,331 ISIC 2019 images, and 656 Stanford DDI images are verified on local drives.
- **Label Compatibility & Exclusions:**
  - HAM10000 maps 1:1 to the seven target classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
  - ISIC 2019 includes an 8th diagnostic category: **`SCC` (Squamous cell carcinoma, 628 images)**, which is incompatible with the 7-class label space and must remain quarantined.
  - Stanford DDI spans 78 granular dermatological conditions. Many are non-neoplastic/inflammatory (e.g. epidermal cyst, verruca vulgaris, mycosis fungoides), and **0 verified normal skin controls exist**.
- **Identifier Availability:**
  - HAM10000 provides complete `image_id` and `lesion_id` for all 10,015 images.
  - ISIC 2019 provides partial `lesion_id` (missing for 2,084 images).
  - Stanford DDI provides **NO patient/subject/lesion identifiers**, which prevents mathematical proof of zero patient-level cross-dataset overlap.
- **Protected Baseline Integrity:** The frozen HAM10000 test partition (1,494 images) remains 100% untouched and isolated from all candidate training sets.

---

## 2. Dataset 1 — HAM10000 (Physical Audit)

- **Official Source:** Kaggle (`kmader/skin-cancer-mnist-ham10000`) / Harvard Dataverse
- **Local Directory:** `D:\DOWNLOADS D\archive`
- **Metadata File:** `D:\DOWNLOADS D\archive\HAM10000_metadata.csv` (563,277 bytes)
- **Image Subdirectories:**
  - `HAM10000_images_part_1`: **5,000 JPEG images**
  - `HAM10000_images_part_2`: **5,015 JPEG images**
  - **Total Physical Image Files:** **10,015 images** (0 unreadable/corrupted)
- **Sample Optical Properties:**
  - Resolution: $600 \times 450$ pixels
  - Color Mode: RGB (24-bit Truecolor)
  - Format: JPEG
- **Patient & Lesion Identifiers:**
  - Total Metadata Rows: **10,015**
  - Unique `image_id`: **10,015** (1:1 mapping with physical files)
  - Unique `lesion_id`: **7,470** (reflects multiple views of identical lesions)
- **Metadata Completeness:**
  - `lesion_id`, `image_id`, `dx`, `dx_type`, `sex`, `localization`: **0 missing values**
  - `age`: 57 missing values (0.57%)
- **Class Distribution:**
  - `nv` (Melanocytic nevus): **6,705** (66.95%)
  - `mel` (Melanoma): **1,113** (11.11%)
  - `bkl` (Benign keratosis-like): **1,099** (10.97%)
  - `bcc` (Basal cell carcinoma): **514** (5.13%)
  - `akiec` (Actinic keratosis): **327** (3.27%)
  - `vasc` (Vascular lesion): **142** (1.42%)
  - `df` (Dermatofibroma): **115** (1.15%)
- **Optical Domain:** Dermoscopy (polarized non-contact and liquid immersion contact plates).
- **Licence:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Permitted for non-commercial educational research.

---

## 3. Dataset 2 — ISIC 2019 (Physical Audit)

- **Official Source:** ISIC 2019 Challenge (`challenge.isic-archive.com/data/`)
- **Local Directory:** `D:\DOWNLOADS D\archive (1)`
- **Metadata Files:**
  - `ISIC_2019_Training_GroundTruth.csv` (1,291,479 bytes; 25,331 rows)
  - `ISIC_2019_Training_Metadata.csv` (1,214,351 bytes; 25,331 rows)
- **Image Directory Structure & Counts on Disk:**
  - `AK/`: **867 images**
  - `BCC/`: **3,323 images**
  - `BKL/`: **2,624 images**
  - `DF/`: **239 images**
  - `MEL/`: **4,522 images**
  - `NV/`: **12,875 images**
  - `SCC/`: **628 images**
  - `VASC/`: **253 images**
  - **Total Physical Image Files:** **25,331 JPEG images** (0 corrupted)
- **Sample Optical Properties:**
  - Resolution: $600 \times 450$ pixels (downsampled challenge standard)
  - Color Mode: RGB (24-bit)
  - Format: JPEG
- **Ground Truth Label Composition (25,331 rows):**
  - `NV`: 12,875 (50.83%)
  - `MEL`: 4,522 (17.85%)
  - `BCC`: 3,323 (13.12%)
  - `BKL`: 2,624 (10.36%)
  - `AK`: 867 (3.42%)
  - `SCC`: **628 (2.48%) — Unsupported 8th class**
  - `VASC`: 253 (1.00%)
  - `DF`: 239 (0.94%)
  - `UNK`: 0
- **Identifier Limitations:**
  - `lesion_id`: Present for 23,247 images; **missing for 2,084 images** (primarily from BCN20000 hospital archives where patient tracking was unlinked).
  - `anatom_site_general`: Missing for 2,631 images.
  - `age_approx`: Missing for 437 images.
- **Label Conflict & Incompatibility Rule:**
  - `SCC` (Squamous cell carcinoma) is a distinct invasive non-melanoma skin cancer. In the 7-class taxonomy, intraepithelial Bowen's disease is grouped into `akiec`, but invasive `SCC` was explicitly excluded from HAM10000.
  - **Rule:** All 628 `SCC` images must be **strictly excluded** from 7-class training manifests to prevent label poisoning.
- **Optical Domain:** Dermoscopy (multi-institutional: BCN20000 Barcelona, HAM10000 Vienna/Queensland, MSK New York).
- **Licence:** CC BY-NC 4.0 (HAM10000, MSK), CC-0 / Custom Research Use (BCN20000). Academic research compliant.

---

## 4. Dataset 3 — Stanford DDI (Physical Audit)

- **Official Source:** Stanford AIMI (`ddi-dataset.github.io`)
- **Local Directory:** `D:\DOWNLOADS D\ddidiversedermatologyimages`
- **Metadata File:** `D:\DOWNLOADS D\ddidiversedermatologyimages\ddi_metadata.csv`
- **Image Files:** **656 PNG images** (`000001.png` to `000656.png`, 227.47 MB)
- **Sample Optical Properties:**
  - Resolution: Variable clinical resolutions (e.g. $948 \times 549$, $1024 \times 768$)
  - Color Mode: RGB (24-bit)
  - Format: PNG
- **Pathology & Clinical Distribution:**
  - Malignancy: **171 True (26.07%)**, **485 False (73.93%)**
  - Normal Skin Controls: **0 images (0.0%)** (all 656 images are biopsy-proven lesions or active complaints).
  - Skin Tone (Fitzpatrick): **FST 12: 208 (31.7%)**, **FST 34: 241 (36.7%)**, **FST 56: 207 (31.6%)** — balanced across light, medium, and dark skin tones.
  - Top Diagnoses: Melanocytic nevi (119), Seborrheic keratosis (58), Verruca vulgaris / warts (50), BCC (41), Epidermal cysts (35), Mycosis fungoides (32), SCC in situ (28), Dermatofibroma (22).
- **Critical Architectural & Ethical Constraints:**
  1. **Missing Patient Identifiers:** Stanford DDI does **NOT** provide `patient_id` or `lesion_id`. It is impossible to prove patient separation within DDI or cross-patient overlap with external web sources.
  2. **Domain Mismatch:** Captured using consumer digital cameras in clinical exam rooms (non-dermoscopic, ambient lighting, macro zoom), representing a completely different optical domain from HAM10000/ISIC dermoscopy.
  3. **Assigned Role:** **100% External Frozen Benchmark**. DDI images must **NEVER** be merged into training or validation splits.
- **Licence:** Stanford AIMI Research-Only Data Use Agreement. Academic research only, strictly no commercial use, strictly no re-identification, no diagnostic clinical use.

---

## 5. Cross-Dataset Overlap, Deduplication & Split Structure

### A. The Inherent HAM10000 $\subset$ ISIC 2019 Overlap
Because HAM10000 was contributed to the ISIC 2019 challenge, a raw union of both datasets duplicates thousands of images:
- **Exact & Near Duplicates:** The deduplication pipeline (`audit_hashes.py`, `src/isic.py`) previously identified:
  - **11,420 duplicate ISIC records** (exact duplicates or near-duplicates within ISIC and with HAM).
  - **3,141 ISIC images that duplicated the HAM10000 test set**!
- **Existing Split Integrity:**
  - Preserved Test Set: **1,494 images** (100% HAM10000, 0 ISIC images).
  - Combined Training: **16,546 images** (6,930 HAM10000 + 9,616 ISIC 2019).
  - Combined Validation: **3,440 images** (1,510 HAM10000 + 1,930 ISIC 2019).
  - **Status:** **Zero image leakage into the preserved test split.**

```
Physical Image Availability Across the Three Datasets:
┌─────────────────────────┬───────────────────┬───────────────────────────────────────┐
│ Dataset Source          │ Physical Images   │ Assigned Architectural Role           │
├─────────────────────────┼───────────────────┼───────────────────────────────────────┤
│ HAM10000                │ 10,015 on disk    │ Baseline Train (70%), Val (15%),      │
│                         │                   │ Frozen Test Benchmark (1,494 images)  │
│ ISIC 2019               │ 25,331 on disk    │ Filtered Supplementary Train & Val    │
│                         │                   │ (Excluding SCC & 11,420 duplicates)   │
│ Stanford DDI            │ 656 on disk       │ 100% External Frozen Benchmark        │
│                         │                   │ (Zero training/validation data)       │
└─────────────────────────┴───────────────────┴───────────────────────────────────────┘
```

---

## 6. Label Taxonomy Mapping Matrix

To safely integrate candidate data into the 7-class lesion space:

| Source Dataset Label | Target 7-Class Label | Clinical Justification | Action |
| :--- | :--- | :--- | :--- |
| **HAM `akiec` / ISIC `AK`** | `akiec` | Actinic keratoses & intraepithelial carcinoma | **Included** |
| **HAM `bcc` / ISIC `BCC`** | `bcc` | Basal cell carcinoma | **Included** |
| **HAM `bkl` / ISIC `BKL`** | `bkl` | Benign keratosis (solar lentigo, seborrheic keratosis) | **Included** |
| **HAM `df` / ISIC `DF`** | `df` | Dermatofibroma | **Included** |
| **HAM `mel` / ISIC `MEL`** | `mel` | Malignant melanoma | **Included** |
| **HAM `nv` / ISIC `NV`** | `nv` | Melanocytic nevus (mole) | **Included** |
| **HAM `vasc` / ISIC `VASC`**| `vasc` | Vascular lesion (angioma, hemangioma) | **Included** |
| **ISIC `SCC`** | **EXCLUDED** | Invasive Squamous Cell Carcinoma (not in 7-class space) | **Quarantined (628 images)** |
| **DDI `malignant` / `benign`**| **EXTERNAL ONLY** | Binary malignancy & 78 granular diseases | **Evaluated externally only** |

---

## 7. Safest Next Experiment Recommendation

Based on this physical audit, the safest, most impactful sequence of next steps is:

### Step 1 (Stage B): Create Versioned Experiment Manifest (`data/experiments/manifest_v1.csv`)
- Construct an isolated manifest linking clean, non-overlapping HAM10000 and ISIC 2019 records.
- Preserve lesion grouping (`lesion_id`). For the 2,084 ISIC records lacking `lesion_id`, treat `image_id` as individual single-sample clusters strictly within the training fold.
- Maintain the **1,494 HAM test set** completely frozen.
- Check collision: Ensure `outputs/splits.csv` and `outputs/combined_splits.csv` remain completely untouched.

### Step 2 (Stage C): Implement Position-Aware Online Augmentation
- In addition to standard horizontal flips, implement on-the-fly **position-aware translations** (sampling left-shifted, center, and right-shifted lesion framing with bounded padding) to improve spatial invariance without offline disk duplication.

### Step 3 (Stage D/E): Execute Controlled Training Experiment
- Combine position-aware augmentation with **Effective Number of Samples (ENS $\beta = 0.999$) class weighting** in `src/train_experiment.py`.
- Evaluate strictly on validation data first.

---

## 8. Stage A Sign-Off & Strict Stop

- **Audit Complete:** Physical audit of HAM10000 (10,015 images), ISIC 2019 (25,331 images), and Stanford DDI (656 images) is fully documented.
- **Safety Compliance:** Zero models retrained, zero files downloaded, zero splits altered, zero protected assets modified.
- **Report Saved:** Saved to [`reports/experiments/three_dataset_audit.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/three_dataset_audit.md).

🛑 **STOPPING FOR YOUR REVIEW AND APPROVAL.**  
Please review this Stage A audit report and let me know if you approve proceeding to **Stage B (Creating the isolated experiment manifest)**.
