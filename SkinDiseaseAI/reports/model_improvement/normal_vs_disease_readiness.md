# 🛑 SkinDiseaseAI — Normal vs. Disease Model Readiness & Blocker Report

**Project:** SkinDiseaseAI — Complete Model Accuracy Improvement and Generalization Project  
**Author:** Senior Machine Learning Engineer & Medical Dataset Quality Specialist  
**Date:** October 29, 2026  
**Document:** `reports/model_improvement/normal_vs_disease_readiness.md`  
**Status:** Binary Classifier Training Formally **BLOCKED** Pending Verified Class 0 Data Acquisition  

---

## 1. Executive Summary & Architectural Separation

A fundamental requirement in medical machine learning is recognizing that **Normal Skin vs. Disease Skin classification is a completely distinct task from 7-class dermoscopic lesion categorization**:
- A seven-class model trained on dermoscopy cannot classify normal skin, smartphone photographs, or everyday non-skin items.
- It was trained strictly on close-ups of pre-selected lesions where the presence of an abnormality was already assumed.
- To safely support consumer smartphone screening, a dedicated **Normal-Looking Skin vs. Possible Skin Disease** binary model must precede any disease-type classification.

```
                    ┌───────────────────────────────────────────────┐
                    │          INCOMING SMARTPHONE IMAGE            │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │         STAGE 1: INPUT SUITABILITY GATE       │
                    │   Rejects non-skin, blur, glare, corruptions  │
                    └───────────────────────┬───────────────────────┘
                                            │
                                  [Passed Suitability]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │     STAGE 2: NORMAL VS. DISEASE CLASSIFIER    │
                    │      (Binary Model — CURRENTLY BLOCKED)       │
                    └───────────────────────┬───────────────────────┘
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                 ┌───────────────────────┐     ┌───────────────────────┐
                 │ CLASS 0: NORMAL SKIN  │     │ CLASS 1: DISEASE SKIN │
                 │ "Normal-looking skin" │     │ "Possible skin lesion"│
                 └───────────────────────┘     └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │ STAGE 3: LESION TYPE  │
                                               │ (7-class model or     │
                                               │  clinical referral)   │
                                               └───────────────────────┘
```

---

## 2. Hard Blocker: Absence of Verified Class 0 Images

> [!CAUTION]
> **PRIMARY BLOCKER: 0 Verified Class 0 Images Currently Exist on Disk**  
> Physical inspection of [`data_normal_disease`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data_normal_disease) confirms that the **Option A volunteer collection (50 consenting subjects) has not yet been executed**.
> 
> Under scientific integrity guidelines:
> 1. **We will NOT fabricate synthetic normal skin.**
> 2. **We will NOT substitute benign lesions** (e.g. moles, warts, or cysts from DDI or HAM10000) as normal skin. Benign lesions are pathological abnormalities, NOT normal unblemished skin.
> 3. **We will NOT use unverified user self-reports** (`LOOKS_HEALTHY` from SCIN).
> 
> **Readiness Verdict:** Training a Normal vs. Disease binary classifier is strictly **BLOCKED** until genuine Class 0 volunteer images are collected under the approved 6-pillar informed-consent protocol.

---

## 3. Candidate Dataset Audit & Assigned Roles

### A. Stanford DDI (Diverse Dermatology Images)
- **Local Path:** `D:\DOWNLOADS D\ddidiversedermatologyimages`
- **Volume:** 656 verified PNG images (227.47 MB).
- **Pathology Composition:** 171 malignant lesions, 485 benign lesions, **0 normal skin controls**.
- **Skin-Tone Diversity:** Fully balanced across Fitzpatrick Skin Types: FST 12 (208), FST 34 (241), FST 56 (207).
- **Assigned Role:** **100% External Frozen Benchmark**.
  - Zero DDI images will ever be used in training or validation.
  - Reserved exclusively to measure external generalization across diverse skin tones once a binary model is trained.
- **Inherent Limitation:** DDI metadata lacks patient identifiers (`patient_id`). Mathematical proof of zero cross-patient overlap with external registries cannot be generated.

### B. Google SCIN (Skin Condition Image Network)
- **Local Path:** `data_normal_disease\scin_repo` (Repository cloned; images reside on GCS/HF).
- **Assigned Role:** Candidate Disease-Skin (Class 1) training source.
- **Filtering Heuristic:**
  - Differential confidence weight $\ge 0.40$ in `weighted_skin_condition_label`.
  - Gradable flag: `dermatologist_gradable_for_skin_condition == True`.
- **Clinical Terminology Standard:** All filtered SCIN samples are formally designated as **"dermatologist differential consensus cases"**, not biopsy-verified histopathology.
- **Case Grouping:** All images sharing a `case_id` must remain strictly grouped within the same partition.

---

## 4. Standardized Manifest Schema (`dataset_manifest.csv`)

To prevent silent relabeling and ensure complete traceability, a 14-column manifest schema is established:

| Column Name | Data Type | Permitted Values / Description |
| :--- | :--- | :--- |
| `image_id` | String | Unique identifier token (e.g. `VOL_001_01`, `DDI_000001`, `SCIN_c102_im1`) |
| `image_path` | String | Path to image file on disk |
| `dataset_source` | String | `OptionA_Volunteer`, `Stanford_DDI`, `Google_SCIN` |
| `subject_id` | String | Unique subject cluster ID (`SUBJ_001`–`SUBJ_050`, `case_id`, `UNAVAILABLE`) |
| `original_label` | String | Verbatim diagnostic label or anatomical region from source metadata |
| `source_malignancy` | String | `malignant`, `benign`, `non_neoplastic`, `unaffected_normal` |
| `candidate_binary_class` | Integer / String | `0` (Normal), `1` (Possible Disease), `EXCLUDED` (Ungradable/OOD) |
| `skin_tone_scale` | String | `FST` (Fitzpatrick 1-6), `MST` (Monk 1-10) |
| `skin_tone_value` | String | Measured tone score (`12`, `34`, `56`, `MST_07`) |
| `anatomical_site` | String | Anatomical location (`forearm_inner`, `hand_dorsum`, `lower_leg`) |
| `lighting_condition` | String | `daylight`, `warm_indoor`, `cool_led`, `clinical_exam` |
| `image_domain` | String | `smartphone_macro`, `smartphone_clinical`, `clinical_camera` |
| `qc_status` | String | `pending_qc`, `passed_qc`, `failed_qc` |
| `role` | String | `training_candidate`, `val_candidate`, `test_candidate`, `external_benchmark_only` |

---

## 5. Unblocking Roadmap & Sequential Next Steps

To transition from the current blocked state to a defensible binary model:

1. **Step 1: Execute Option A Volunteer Collection**
   - Collect consented normal skin photographs across 50 independent subjects (~900–950 usable images post-QC).
   - Ensure documented ethical consent and participant anonymization.
2. **Step 2: Authorize & Download SCIN Consensus Subset**
   - Download the filtered Google SCIN differential consensus subset (~5,000–6,000 images).
3. **Step 3: Construct Leak-Proof Subject-Clustered Splits**
   - Partition 70% Train, 15% Validation, 15% Internal Test clustered strictly by `subject_id` / `case_id`.
   - Verify zero SHA-256 and pHash duplicates across splits.
4. **Step 4: Train Baseline Binary EfficientNetB0**
   - Execute 2-phase training (Frozen backbone $\to$ top 30 layers fine-tuning).
   - Tune classification threshold $t^*$ on Validation split targeting $\ge 85\%$ disease sensitivity.
5. **Step 5: External Evaluation on Stanford DDI**
   - Evaluate ONCE on the frozen internal test split and the external Stanford DDI benchmark.
