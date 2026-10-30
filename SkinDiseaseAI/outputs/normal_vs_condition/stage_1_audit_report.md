# 🔬 Stage 1 — Imported Datasets Audit: Normal Skin vs. Visible Skin Condition

**Project:** SkinDiseaseAI — Normal-Looking Skin vs. Visible Skin Condition Screening  
**Role:** Senior Machine Learning Engineer, Deep Learning Researcher & Medical CV Specialist  
**Date:** October 29, 2026  
**Document:** `outputs/normal_vs_condition/stage_1_audit_report.md`  
**Execution Environment:** Windows 11 | Python 3.12.10 | 24 GB RAM | NVIDIA RTX 3050  
**Status:** **Stage 1 Complete (Inspection, Shortcut Audit & Manifest Only — Zero Models Trained)**

---

## 1. Executive Summary & Critical Clinical Finding

> [!CRITICAL]
> **Foundational Finding: Ground-Truth Normal Skin Absence**  
> Across all three imported sources (**Stanford DDI**, **Fitzpatrick17k**, and **Google SCIN**), exactly **ZERO verified pathologist- or physician-certified normal/healthy skin control images exist locally on disk**.
>
> 1. **Stanford DDI (656 images on disk):** 100% of images are biopsy-confirmed malignant (171) or benign (485) lesions. Benign lesions (seborrheic keratoses, warts, moles, cysts) are active dermatological lesions and are **NOT normal skin**.
> 2. **Fitzpatrick17k (16,577 records):** Cloned metadata repository contains URLs to online dermatology disease atlases. **0 image files are stored locally.** Querying for normal or healthy controls yields **0 records**.
> 3. **Google SCIN (5,033 cases, 10,407 photos):** Cloned repository contains code and schemas. Physical image files remain remote (12.6 GB on Google Cloud Storage / Hugging Face). User-reported `LOOKS_HEALTHY` is an unverified layperson self-perception, not medical ground truth.

---

## 2. Source-by-Source Technical & Legal Audit Table

| Audit Parameter | Stanford DDI (`ddidiversedermatologyimages.zip`) | Fitzpatrick17k (`fitzpatrick17k_repo`) | Google SCIN (`scin_repo`) |
| :--- | :--- | :--- | :--- |
| **Local Path** | `D:\DOWNLOADS D\ddidiversedermatologyimages` | `data_normal_disease\fitzpatrick17k_repo` | `data_normal_disease\scin_repo` |
| **License & Terms** | Stanford AIMI Data Use Agreement (Research-only, non-commercial, no re-identification) | Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0) | Google Research SCIN Open Data License (Non-commercial research & education) |
| **Physical Images on Disk** | **656 PNG images** (`000001.png`–`000656.png`) | **0 image files** (metadata CSV with web URLs only) | **0 image files** (12.6 GB hosted remotely on GCS/HF) |
| **Total Metadata Rows** | **656 rows** (`ddi_metadata.csv`) | **16,577 rows** (`fitzpatrick17k.csv`) | **5,033 cases** (10,407 photos in published release) |
| **Label Scheme** | 78 granular disease entities + binary `malignant` (True: 171, False: 485) | 114 specific diseases; 9-partition labels; 3-partition labels (`non-neoplastic`, `malignant`, `benign`) | Dermatologist differential diagnoses with confidence scores (1–5) and weighted condition labels |
| **Patient Identifiers** | **None** (Image-level index only) | **None** (Scraped from web dermatology atlases) | **`case_id` provided** (groups 1–3 photos per patient encounter) |
| **Body Location Metadata** | **Unspecified** (No anatomical column; only 9 acral/hand diagnoses) | **Unspecified** (No anatomical column; inferred for 573 hand-specific conditions) | **Explicit boolean columns** (`body_parts_back_of_hand`, `body_parts_palm`, etc.) |
| **Skin Tone Scale** | Grouped Fitzpatrick: FST 12 (208), FST 34 (241), FST 56 (207) | Fitzpatrick Scale: FST 1–6 (Type 1: 2947, 2: 4808, 3: 3308, 4: 2781, 5: 1533, 6: 635, Unk: 565) | Estimated Monk Skin Tone (eMST 1–10) & self-reported Fitzpatrick (FST 1–6) |
| **Image Dimensions** | Variable clinical macro: $500\times680$ to $1031\times1401$ px | Variable web atlas resolutions (not downloaded) | Smartphone resolutions (typically $1080\times1440$ to $3024\times4032$ px) |
| **Corrupt / Unreadable Files** | **0 corrupt files** (656/656 decoded via PIL) | N/A (0 images stored locally) | N/A (0 images stored locally) |
| **Verified Normal Skin Controls**| **0 images (0.0%)** | **0 records (0.0%)** | **0 verified controls (0.0%)** |

---

## 3. Hand & Finger Anatomical Location & Condition Audit

### A. Anatomical Location Breakdown Across Sources

| Anatomical Site | Stanford DDI (on disk) | Fitzpatrick17k (metadata records) | Google SCIN (official case metadata) |
| :--- | :---: | :---: | :---: |
| **Back of Hand** | *Unspecified in metadata* | *Inferred via condition/URL* | **521 cases** |
| **Palm** | *Unspecified in metadata* | *Inferred via condition/URL* | **215 cases** |
| **Total Hand / Finger Cases** | **9 (Acral-specific)** | **573 (Hand/acral conditions)** | **736 cases (~1,500+ photos)** |
| **Arm** | *Unspecified* | *Unspecified* | 1,707 cases |
| **Leg** | *Unspecified* | *Unspecified* | 1,694 cases |
| **Torso (Front / Back)** | *Unspecified* | *Unspecified* | 1,447 cases |
| **Head or Neck** | *Unspecified* | *Unspecified* | 737 cases |
| **Foot (Sole / Top)** | *Unspecified* | *Unspecified* | 519 cases |
| **Other / Unspecified** | 647 images | 16,004 records | 543 cases |

### B. Specific Conditions Documented on Hands / Fingers

| Condition Entity | Dataset Source | Specific Hand / Acral Manifestation | Image / Case Count |
| :--- | :--- | :--- | :---: |
| **Allergic / Irritant Contact Dermatitis** | Google SCIN | Dorsal hands and palms (occupational / chemical exposure) | Top 3 hand condition |
| **Dyshidrotic Eczema (Pompholyx)** | Fitzpatrick17k & SCIN | Pruritic tapioca-like vesicles on lateral fingers & palms | 83 (Fitz) + frequent in SCIN |
| **Scabies** | Fitzpatrick17k | Burrows and papules in finger web spaces and wrists | 339 (Fitz) |
| **Acrodermatitis Enteropathica** | Fitzpatrick17k | Periorificial and acral erosive dermatitis | 92 (Fitz) |
| **Paronychia** | Fitzpatrick17k | Periungual inflammation around finger/toe nail folds | 59 (Fitz) |
| **Melanoma (Acral Lentiginous)** | Stanford DDI | Malignant melanomas on palms, soles, or under nails | 7 (DDI) |
| **Focal Acral Hyperkeratosis** | Stanford DDI | Keratotic papules along borders of hands and feet | 1 (DDI) |
| **Acral Melanotic Macule** | Stanford DDI | Benign pigmented macule on hand/foot | 1 (DDI) |
| **Verruca Vulgaris (Common Warts)** | DDI & SCIN | Hyperkeratotic papules on periungual and dorsal fingers | 50 in DDI (site unlinked) |

---

## 4. Shortcut Learning Audit: Mathematical Predictability

A rigorous empirical shortcut audit was conducted to test whether trivial, non-pathological features (file size, image width, image height, aspect ratio, image format, mean R/G/B channels, and channel standard deviations) can trivially predict the **dataset source** or **diagnostic label**:

### Shortcut Test Matrix (5-Fold Stratified Cross-Validation):

```
┌───────────────────────────────────────────────┬──────────────────────┬──────────────────────┬─────────────┬───────────────────────────┐
│ Dataset Comparison / Prediction Task          │ Logistic Regression  │ Decision Tree        │ ROC-AUC     │ Shortcut Confounding Flag │
├───────────────────────────────────────────────┼──────────────────────┼──────────────────────┼─────────────┼───────────────────────────┤
│ DDI vs. HAM10000 (Predicting Image Source)   │ 97.56% ± 1.01%       │ 100.00% ± 0.00%      │ 0.9950      │ 🚨 CRITICAL SHORTCUT RISK │
│ DDI vs. ISIC 2019 (Predicting Image Source)  │ 96.57% ± 2.94%       │ 100.00% ± 0.00%      │ 0.9869      │ 🚨 CRITICAL SHORTCUT RISK │
│ DDI vs. HAM10000 (Geometry & Color Only)     │ 96.49% ± 0.88%       │ 99.85% ± 0.20%       │ 0.9912      │ 🚨 CRITICAL SHORTCUT RISK │
│ Within DDI (Predicting Malignant vs. Benign) │ 75.15% ± 0.77%       │ 72.71% ± 1.27%       │ 0.6492      │ 🟢 Safe (Near 73.9% Prior)│
└───────────────────────────────────────────────┴──────────────────────┴──────────────────────┴─────────────┴───────────────────────────┘
```

### Critical Shortcut Findings:
1. **100% Decision Tree Predictability (DDI vs. HAM/ISIC):**  
   A shallow 3-depth decision tree achieves **100.0% accuracy** distinguishing Stanford DDI clinical images from HAM10000 dermoscopy images using only file format (PNG vs. JPEG), image aspect ratio, and mean color.
2. **Color & Geometry Alone Achieves 96.5% Separation:**  
   Even when stripping file format headers, the physical image dimensions, aspect ratio, and mean background color separate DDI from HAM10000 with **96.49% accuracy**.
3. **Severe Confounding Warning:**  
   If Stanford DDI (or external clinical photos) is combined naively with dermoscopy images, a deep learning model will achieve $>98\%$ apparent accuracy **without learning a single pathological feature of skin disease**. It will simply classify the camera sensor, optical zoom, and image dimensions.

---

## 5. Clinical Manifest Construction: `manifest_clinical_v1.csv`

In accordance with Stage 1, an isolated manifest was constructed under the designated new directory:
- **Manifest Path:** [`data_clinical/manifest_clinical_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data_clinical/manifest_clinical_v1.csv)
- **Total Records Cataloged:** **17,233 records**
  - **Stanford DDI (Physical Images on Disk):** **656 records** (`split`: `external_eval`, `file_exists`: `True`)
  - **Fitzpatrick17k (Metadata URLs Only):** **16,577 records** (`split`: `excluded`, `file_exists`: `False`)
- **Isolation Compliance:**
  - `models/normal_disease/` remains **completely empty**.
  - Protected files (`config.json`, `models/*.keras`, `outputs/splits.csv`, `outputs/combined_splits.csv`) remain **100% untouched**.
  - Zero external web scraping or unapproved downloads were performed.

---

## 6. Stage 1 Sign-Off & Strict Stop

- **Audit Complete:** Physical audit of Stanford DDI (656 images on disk), Fitzpatrick17k (16,577 remote records), and SCIN (5,033 cases, 736 hand cases) is complete.
- **Shortcut Evidence:** 100% non-medical separability between DDI and dermoscopy proven empirically.
- **Normal Skin Reality:** **0 verified normal skin controls exist across all three imported sources.**
- **Manifest Saved:** [`data_clinical/manifest_clinical_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data_clinical/manifest_clinical_v1.csv).

🛑 **STAGE 1 COMPLETE — STOPPING FOR YOUR REVIEW AND APPROVAL.**  
Please review the audit tables and shortcut learning findings above before authorizing **Stage 2 (Pipeline Design & Normal-Skin Volunteer Collection Plan)**.
