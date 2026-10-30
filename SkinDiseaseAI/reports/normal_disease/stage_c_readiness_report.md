# 📊 SkinDiseaseAI — Stage C: Dataset Preparation and Readiness Report
**Project:** SkinDiseaseAI — Normal-Looking Skin vs. Possible Skin Disease Classifier  
**Role:** Senior Machine Learning Engineer (Medical Computer Vision & Dataset Quality)  
**Date:** October 29, 2026  
**Status:** Stage C Readiness Check Complete (Audit & Readiness Checks Only — No Training)

---

## 1. Project Integrity Verification

All locked and frozen assets in the existing `SkinDiseaseAI` project were inspected programmatically. All file sizes, timestamps, and hashes remain **100% intact and unmodified**:

| Asset / File Path | Status on Disk | Verified Size (Bytes) | Last Modified Timestamp | Integrity Check |
| :--- | :--- | :--- | :--- | :--- |
| `models\best_model.keras` | **EXISTS** | 33,486,076 | 2026-10-15 21:50:51 | **Preserved / Unchanged** |
| `models\isic_augmented_v1.keras` | **EXISTS** | 33,477,861 | 2026-10-26 12:18:27 | **Preserved / Unchanged** |
| `models\isic_finetuned.keras` | **EXISTS** | 33,477,534 | 2026-10-23 00:39:24 | **Preserved / Unchanged** |
| `outputs\splits.csv` | **EXISTS** | 1,259,807 | 2026-10-15 21:23:43 | **Preserved / Unchanged** |
| `outputs\combined_splits.csv` | **EXISTS** | 2,556,961 | 2026-10-23 00:11:18 | **Preserved / Unchanged** |
| `config.json` | **EXISTS** | 1,985 | 2026-10-26 15:46:08 | **Preserved / Unchanged** |
| `D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip` | **EXISTS** | 1,506,237,677 | 2026-10-27 10:10:21 | **Locked Master Backup Intact** |

*No existing models were retrained, no thresholds altered, and no existing reports modified.*

---

## 2. Actual Available Data Audit (DDI and SCIN)

### A. Stanford DDI (Diverse Dermatology Images)
- **Local Path:** `D:\DOWNLOADS D\ddidiversedermatologyimages`
- **Physical Verification:** Exactly 656 PNG image files (`000001.png` to `000656.png`) and `ddi_metadata.csv` (656 rows) are verified on disk.
- **Image Integrity:** 0 corrupted images. Total disk space: **227.47 MB**.
- **Pathology / Labels:** 
  - Malignant: 171 images (melanoma, BCC, SCC, cutaneous lymphoma).
  - Benign lesions: 485 images (nevi, seborrheic keratosis, warts, cysts, dermatofibroma).
  - Normal skin controls: **0 images**.
- **Skin-Tone Distribution:** FST 12 (208), FST 34 (241), FST 56 (207) — fully balanced across skin tones.
- **License / Terms:** Stanford AIMI Research-Only Data Use Agreement (academic non-commercial use only, no re-identification, no patient care).
- **Approved Role:** **100% External Frozen Benchmark** (zero images used in training/validation).

### B. Google SCIN (Skin Condition Image Network)
- **Local Path:** `data_normal_disease\scin_repo`
- **Repository Metadata Status:** Cloned locally (contains `dataset_schema.md`, `README.md`, `scin_demo.ipynb`).
- **Image Collection Status:** **Not yet downloaded to local disk.** The 12.6 GB dataset resides on Google Cloud Storage (`gs://dx-scin-public-data`) and Hugging Face (`google/scin`).
- **Governance:** Governed by Google's SCIN Data Use License (open access for research/education). GCS access requires GCP project authentication; Hugging Face access requires streaming or downloading via `datasets`.
- **Labels:** Dermatologist differential confidence weights (`weighted_skin_condition_label`).

---

## 3. Class 0 (Normal Skin) Status: PRIMARY BLOCKER IDENTIFIED

> [!CAUTION]
> **Primary Blocker: 0 Verified Class 0 Images Currently Exist**  
> Physical audit of `data_normal_disease` confirms that the **Option A volunteer collection (50 consenting subjects) has not yet been executed**.
> 
> In accordance with scientific integrity guidelines:
> - **We will NOT fabricate synthetic normal skin.**
> - **We will NOT substitute benign lesions** (e.g. moles, warts, or cysts from DDI or HAM10000) as normal skin.
> - **We will NOT use unverified user self-reports** (`LOOKS_HEALTHY` from SCIN).
> 
> **Readiness Verdict:** Training a Normal vs. Disease classifier is **BLOCKED** until genuine Class 0 volunteer images are collected under the approved 6-pillar informed-consent protocol.

---

## 4. SCIN Case Grouping & Filtering Rule Verification

Inspection of `dataset_schema.md` and `scin_demo.ipynb` confirms:
1. **Case-Level Clustering:**
   - Every contributor submission is indexed by a unique `case_id`.
   - Each case contains up to 3 photographs (`image_1_path`, `image_2_path`, `image_3_path`) representing close-up, angle, and distance views of the same condition.
   - All images sharing a `case_id` must remain grouped together to prevent patient-level leakage across splits.
2. **Proposed Filtering Heuristic:**
   - Top differential diagnosis weight in `weighted_skin_condition_label` $\ge 0.40$.
   - `dermatologist_gradable_for_skin_condition == True`.
3. **Clinical Terminology Standard:**
   - Cases passing this heuristic represent **"dermatologist differential consensus cases"**, not biopsy-proven pathology.

---

## 5. Proposed Manifest Schema & Eligible Counts

To track all images across repositories without silent relabeling, the following standardized manifest schema is proposed:

### A. Manifest Schema Definition (`dataset_manifest.csv`)

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `image_id` | String | Unique image token (e.g., `VOL_001_01`, `DDI_000001`, `SCIN_c102_im1`) |
| `image_path` | String | Path to physical image file |
| `dataset_source` | String | Source repository (`OptionA_Volunteer`, `Stanford_DDI`, `Google_SCIN`) |
| `subject_id` | String | Subject/case cluster ID (`SUBJ_001`–`SUBJ_050`, `case_id`, `UNAVAILABLE`) |
| `original_label` | String | Verbatim diagnostic label or anatomical site from source |
| `source_malignancy` | String | `malignant`, `benign`, `non_neoplastic`, `unaffected_normal` |
| `candidate_binary_class` | Integer | `0` (Normal), `1` (Possible Disease), `EXCLUDED` (Ungradable/OOD) |
| `skin_tone_scale` | String | `FST` (Fitzpatrick 1-6), `MST` (Monk 1-10) |
| `skin_tone_value` | String | Specific score (e.g., `12`, `34`, `56`, `MST_07`) |
| `anatomical_site` | String | Body site (e.g., `forearm_inner`, `hand_dorsum`, `lower_leg`) |
| `lighting_condition` | String | `daylight`, `warm_indoor`, `cool_led`, `clinical_exam` |
| `image_domain` | String | `smartphone_macro`, `smartphone_clinical`, `clinical_camera` |
| `qc_status` | String | `pending_qc`, `passed_qc`, `failed_qc` |
| `role` | String | `training_candidate`, `val_candidate`, `test_candidate`, `external_benchmark_only` |

### B. Summary of Eligible Candidate Images by Source

| Dataset Source | Local Availability | Eligible Class 0 (Normal) | Eligible Class 1 (Disease) | Proposed Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| **Option A Volunteers** | **Not yet collected (BLOCKER)** | Planned **~900–950** (50 subjects) | 0 | Primary Train (70%), Val (15%), Internal Test (15%) |
| **Stanford DDI** | **656 Verified on Disk** | 0 | **656** (171 mal, 485 ben) | **100% External Frozen Benchmark** |
| **Google SCIN** | **External on HF/GCS** | 0 | Estimated **~5,000–6,000** | Candidate Disease Train (70%), Val (15%), Test (15%) |
| **Fitzpatrick17k** | **Metadata only (0 images)** | 0 | Quarantined | Excluded from Phase 1 training |

---

## 6. Duplicate Auditing & Missing Patient ID Limitations

1. **Intra-Dataset Deduplication Plan:**
   - Exact hash deduplication: SHA-256 byte-level hash computed for every candidate image.
   - Near-duplicate detection: Perceptual hashing (pHash) with Hamming distance threshold $\le 4$.
2. **Cross-Dataset Duplicate & Overlap Constraints:**
   - DDI images were collected at Stanford outpatient dermatology clinics.
   - HAM10000 images were collected at the Medical University of Vienna and the University of Queensland.
   - SCIN images were contributed online by US internet users.
   - **Inherent Documentation Limitation:** Because Stanford DDI lacks patient identifiers, mathematical proof of zero patient overlap with external web atlases cannot be generated. This limitation is formally recorded as an inherent constraint of the DDI dataset.

---

## 7. Input-Suitability Gate Review & Recalibration Plan

### A. Audit of Existing Gate Thresholds
- The existing gate (`src/gate.py`, `models/ood_gate.npz`) employs:
  - Mahalanobis Distance threshold: `thr = 2673.08`
  - Laplacian blur threshold: `blur = 40.0`
  - Luminance limits: `lum_min = 60.0`
- **Finding:** The Mahalanobis centroid $\mu$ and covariance matrix in `ood_gate.npz` were fitted strictly on **dermoscopy images** from HAM10000/ISIC.
- **Why this fails on smartphone images:** Non-contact smartphone photos lack dermatoscope contact plates and peripheral vignette rings. A genuine smartphone photo of healthy skin produces an out-of-distribution distance ($d \approx 5755 > 2673$), causing false rejection!

### B. Recalibration Plan for Smartphone Suitability Gate
Before deploying the suitability gate on smartphone images:
1. **Empirical Evaluation Cohort:**
   - Positive validation set: Real smartphone photos of skin (from Option A volunteers).
   - Negative validation set: 100 real non-skin unrelated images (clothing, room interiors, furniture, documents, pets).
2. **Acceptance Criteria:**
   - False Rejection Rate (FRR) on genuine smartphone skin: $\le 5.0\%$.
   - False Acceptance Rate (FAR) on non-skin objects: $\le 10.0\%$.
3. **Separation of Concerns:** The suitability gate acts strictly as an **input admission filter**. It never predicts disease categories.

---

## 8. Hardware, Memory & Training Estimates

### A. System Profile (Measured on Host Machine)
- **Host OS:** Windows 11 (64-bit).
- **Physical RAM:** **24.0 GB**.
- **Available Free Disk Space on Drive D:** **182.66 GB Free** (Disk capacity is abundant; full SCIN + volunteer images require $<15$ GB).
- **GPU Status:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
  - *Runtime Constraint:* Native Windows TensorFlow $\ge 2.11$ does not support native GPU acceleration and executes via multithreaded CPU instructions (AVX2/oneDNN).

### B. Training Runtime & Memory Estimates (EfficientNetB0)
- **Input Dimensions:** $224 \times 224 \times 3$.
- **Batch Size:** 32.
- **Projected Training Volume:** ~1,800 images (900 Normal + 900 Disease).
- **Throughput on CPU:** ~35–45 images/second.
- **Epoch Duration:** ~45 seconds per epoch.
- **Phased Training Time:**
  - Phase 1 (Frozen Backbone, 10 epochs): **~7.5 minutes**.
  - Phase 2 (Fine-tuning top 30 layers, 10 epochs): **~11.0 minutes**.
  - **Total Estimated Training Time:** **~18.5 minutes** (safely within the 20-minute user threshold).
- **RAM Headroom:** Peak memory during batch streaming via `tf.data.Dataset` will be **~3.8 GB**, well within the system's 24 GB physical RAM.

---

## 9. Comprehensive Readiness Checklist & Actionable Next Steps

| Pre-Training Milestone | Current Status | Required Action to Unblock |
| :--- | :--- | :--- |
| **Asset Integrity Verification** | **COMPLETE & VERIFIED** | None (Existing models & splits 100% locked) |
| **Stanford DDI Local Availability** | **COMPLETE & VERIFIED** | 656 images verified; locked as External Benchmark |
| **Option A Volunteer Normal Skin** | **NOT YET COLLECTED (BLOCKER)** | Execute consented volunteer collection (50 subjects) |
| **Google SCIN Local Images** | **EXTERNAL ON HF/GCS** | Plan targeted consensus download (upon approval) |
| **Smartphone Suitability Gate** | **RECALIBRATION REQUIRED** | Calibrate threshold on real phone skin vs non-skin |
| **Manifest & Split Construction** | **SCHEMA PROPOSED** | Awaiting Class 0 data collection |

---

## 10. Stage C Sign-off & Stop

- **Artifact Created:** Saved to [`reports/normal_disease/stage_c_readiness_report.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/normal_disease/stage_c_readiness_report.md).
- **Compliance:** Zero volunteer images collected, zero datasets downloaded, zero splits created, zero models trained, and zero project files altered.
- **Stage Status:** **Stage C readiness check complete.** Awaiting your review and explicit instructions before proceeding with data acquisition or collection.
