# 🔬 Stage A — Comprehensive Dataset Audit Report: SCIN, Fitzpatrick17k & Stanford DDI
**Project:** SkinDiseaseAI — Normal Skin vs. Disease Skin Classification  
**Role:** Senior Machine Learning Engineer (Medical Computer Vision & Dataset Quality)  
**Date:** October 29, 2026  
**Status:** Stage A Complete (Inspection Only — No Training)

---

## 1. Executive Summary & Critical Clinical Finding

> [!CRITICAL]
> **Foundational Finding: The Normal Skin Ground-Truth Absence**  
> **None of the three candidate dermatology datasets (Stanford DDI, Fitzpatrick17k, or SCIN) contains verified, pathologist-annotated Normal / Healthy Skin control images.**  
>
> All three datasets were explicitly created by dermatologists and medical researchers to study **active dermatological diseases, neoplasms, lesions, and inflammatory conditions** across diverse skin tones.
>
> - **DDI (656 images):** 100% of images are confirmed benign (485) or malignant (171) skin lesions. **0 normal skin controls.**
> - **Fitzpatrick17k (16,577 records):** 100% of images are dermatological conditions from clinical atlases (`non-neoplastic`: 12,080, `malignant`: 2,263, `benign`: 2,234). **0 normal skin controls.**
> - **SCIN (10,000+ images):** Crowdsourced from individuals searching for skin, hair, or nail complaints. 97.9% are physician-confirmed active conditions. User-reported `LOOKS_HEALTHY` is an unverified layperson self-report, and cases marked `gradable = False` represent ungradable/ambiguous photos rather than clinical normal skin controls.
>
> **Clinical Warning against Shortcut Learning:**  
> A benign mole, seborrheic keratosis, wart, or cyst is **NOT** normal skin. If a model is trained by labelling benign lesions as "Class 0: Normal", the model will learn that having pigmented lesions, warts, or cysts is healthy skin, while completely failing on clean, unblemished, healthy skin. Furthermore, pulling "normal skin" from uncontrolled social-media selfies or non-medical datasets would induce catastrophic dataset confounding (the model would learn camera sensors, lighting, and demographic backgrounds rather than pathological features).

---

## 2. Dataset 1 — Stanford DDI (Diverse Dermatology Images)

### A. Dataset Overview & Local Path
- **Local Folder:** `D:\DOWNLOADS D\ddidiversedermatologyimages`
- **Metadata File:** `D:\DOWNLOADS D\ddidiversedermatologyimages\ddi_metadata.csv` (30,655 bytes)
- **Archive Status:** Fully extracted and verified locally.

### B. Image File Audit & Integrity
- **Total Referenced in Metadata:** 656 images (`000001.png` to `000656.png`)
- **Total Physical PNG Files on Disk:** 656 images (**100% match**)
- **Corrupted / Unreadable Images:** **0** (All 656 decoded and verified via PIL)
- **Total Disk Size:** **227.47 MB**
- **File Size Distribution:** Min: 20.6 KB | Max: 1,413.3 KB | Mean: 355.1 KB
- **Color Channels / Modes:** 
  - `RGB` (3 channels): 428 images (65.2%)
  - `RGBA` (4 channels with alpha channel): 228 images (34.8%)
- **Image Dimensions:** Variable high-resolution macro clinical photography (e.g. 500×680, 525×701, 790×636, 932×1252, 1019×1385, 1031×1401).

### C. Annotations, Pathology & Skin-Tone Distribution
- **Metadata Columns:** `['Unnamed: 0', 'DDI_ID', 'DDI_file', 'skin_tone', 'malignant', 'disease']`
- **Patient Identifiers:** **None available** in metadata.
- **Malignancy Breakdown:**
  - `malignant == True`: 171 images (26.1%)
  - `malignant == False`: 485 images (73.9%) — *All 485 are benign lesions, NOT normal skin*
- **Skin-Tone Distribution (Fitzpatrick Grouped Scale):**
  - `skin_tone == 12` (FST I–II): 208 images (31.7%)
  - `skin_tone == 34` (FST III–IV): 241 images (36.7%)
  - `skin_tone == 56` (FST V–VI): 207 images (31.6%)
  *(Exceptionally balanced across dark and light skin tones).*
- **Diagnostic Categories:** 78 unique disease entities.
  - Top conditions: `melanocytic-nevi` (119), `seborrheic-keratosis` (58), `verruca-vulgaris` (50), `basal-cell-carcinoma` (41), `epidermal-cyst` (35), `mycosis-fungoides` (32), `squamous-cell-carcinoma-in-situ` (28), `dermatofibroma` (22), `acrochordon` (19), `squamous-cell-carcinoma` (17), `dysplastic-nevus` (16).

### D. Imaging Domain, License & Usage Restrictions
- **Domain:** Clinical photography (standard digital cameras in dermatology examination rooms; not dermoscopy).
- **License / Agreement:** Stanford AIMI Research-Only Data Use Agreement.
  - Strictly limited to non-commercial academic research.
  - Prohibits re-identification and clinical diagnostic deployment on patients.
- **Suitability:** **Gold-standard for external OOD testing of clinical disease lesions across skin tones**, but cannot provide the Normal Skin class.

---

## 3. Dataset 2 — Fitzpatrick17k

### A. Repository & Metadata Inspection
- **Repository:** `https://github.com/mattgroh/fitzpatrick17k.git`
- **Local Inspection Folder:** `data_normal_disease\fitzpatrick17k_repo`
- **Metadata File:** `fitzpatrick17k.csv` (16,577 rows, 4,120,919 bytes)
- **Image File Availability:** **0 image files in the repository.** The repository only provides metadata with image URLs pointing to external dermatology atlases (*Atlas Dermatologico* and *DermaAmin*).
- **URL Status:** Many external links are broken/defunct. Full archive access requires submitting an author request form (Google Form).

### B. Label Distribution & Hierarchy
- **Metadata Columns:** `['md5hash', 'fitzpatrick_scale', 'fitzpatrick_centaur', 'label', 'nine_partition_label', 'three_partition_label', 'qc', 'url', 'url_alphanum']`
- **Three-Partition Labels:**
  - `non-neoplastic`: 12,080 (72.9%) — inflammatory, infectious, autoimmune, and genetic diseases
  - `malignant`: 2,263 (13.7%) — carcinomas, melanomas, lymphomas
  - `benign`: 2,234 (13.5%) — benign tumors, cysts, epidermal growths
- **Nine-Partition Labels:**
  - `inflammatory`: 10,886 | `malignant epidermal`: 1,352 | `genodermatoses`: 1,194 | `benign dermal`: 1,067 | `benign epidermal`: 931 | `malignant melanoma`: 573 | `benign melanocyte`: 236 | `malignant cutaneous lymphoma`: 182 | `malignant dermal`: 156
- **Normal Skin Records:** **0 records.** Querying for normal/healthy/control labels yields exactly 0 hits.
- **Patient Identifiers:** **None.** Scraped from web atlases without patient IDs.

### C. Fitzpatrick Skin-Type Representation
- **Scale:** Fitzpatrick I to VI:
  - FST 1: 2,947 (17.8%)
  - FST 2: 4,808 (29.0%)
  - FST 3: 3,308 (20.0%)
  - FST 4: 2,781 (16.8%)
  - FST 5: 1,533 (9.2%)
  - FST 6: 635 (3.8%)
  - Unknown (-1): 565 (3.4%)
  *(Heavy skew toward lighter skin types: Types I–III represent 66.8% of the data; Type VI represents only 3.8%).*

### D. Imaging Domain & License
- **Domain:** Clinical photography scraped from web atlases.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0).
- **Suitability:** Valuable for diverse clinical disease pretraining and inflammatory condition coverage, but provides 0 normal controls.

---

## 4. Dataset 3 — SCIN (Skin Condition Image Network)

### A. Repository & Data Source
- **Repository:** `https://github.com/google-research-datasets/scin.git`
- **Local Inspection Folder:** `data_normal_disease\scin_repo`
- **Official Dataset Hosting:** Hugging Face (`google/scin`) and Google Cloud Storage (`gs://dx-scin-public-data`).
- **Total Download Size:** **12.6 GB** (Parquet / image archives).
- **Access / Terms:** Governed by Google's SCIN Data Use License (open access for research/education). GCS access requires GCP project authentication; Hugging Face access requires streaming/downloading via the `datasets` library.

### B. Dataset Composition & Annotations
- **Submissions:** 5,749 contributions, ~10,000+ smartphone photos.
- **Clinical Nature:** 97.9% genuine dermatology conditions (allergies, infections, eczema, psoriasis, acne).
- **Physician Ground Truth:** Up to 3 dermatologists per case providing differential diagnoses with confidence scores (1–5), aggregated into `weighted_skin_condition_label`.
- **Can SCIN provide "Normal Skin"?**
  - In `scin_cases.csv`, `related_category` contains `LOOKS_HEALTHY`, but this represents **user self-perception**, not medical confirmation.
  - In `scin_labels.csv`, `dermatologist_gradable_for_skin_condition == False` represents ungradable images (poor focus, dark lighting, conflicting lesions, or absence of discernible pathology), **not verified healthy controls**.
  - **Pathologist- or physician-certified normal skin controls: 0.**

### C. Skin Tone & Demographic Diversity
- **Estimated Monk Skin Tone (eMST 1–10):** Graded by trained panels in India and the US.
- **Estimated Fitzpatrick Skin Type (eFST 1–6):** Dermatologist estimated.
- **Self-Reported Fitzpatrick Skin Type (sFST):** User-reported sun reactivity.
- **Patient Identifiers:** `case_id` provided (1 to 3 images per case: close-up, angle, distance).
- **Suitability:** High-value benchmark for consumer smartphone photography of skin infections and rashes, but lacks verified healthy skin controls.

---

## 5. Cross-Dataset Comparison Matrix

| Property | Stanford DDI | Fitzpatrick17k | Google SCIN | HAM10000 / ISIC 2019 (Existing) |
| :--- | :--- | :--- | :--- | :--- |
| **Local Availability** | **Extracted & Verified** (`D:\DOWNLOADS D`) | Metadata cloned; Images via URL/Form | Metadata cloned; 12.6 GB on HF/GCS | Fully processed in workspace |
| **Total Images** | 656 | 16,577 | ~10,000+ | 10,015 (HAM) / 25,331 (ISIC) |
| **Imaging Domain** | Clinical macro photography | Clinical atlas photography | Smartphone user photography | **Dermoscopy only** |
| **Normal Skin Count** | **0** | **0** | **0 (unverified)** | **0** (All 7 classes are lesions) |
| **Benign Lesion Count** | 485 | 2,234 | Diffuse | 8,229 `nv`, 1,746 `bkl`, 173 `df` |
| **Malignant Count** | 171 | 2,263 | Variable | 3,138 `mel`, 2,396 `bcc`, 690 `akiec` |
| **Skin Tone Metadata** | FST 12, 34, 56 (Balanced) | FST I–VI (Light skew) | Monk 1–10 & FST (Broad) | None / unannotated |
| **Patient IDs** | None in metadata | None | `case_id` present | `lesion_id` / `patient_id` present |
| **License** | Stanford AIMI Research Agreement | CC BY-NC-SA 3.0 | SCIN Research License | CC-BY-NC 4.0 / Creative Commons |

---

## 6. Scientific Solutions for the Normal Skin Class

Because none of the three datasets contains verified healthy skin, building a genuine Normal vs. Disease model requires an explicit strategy for **Class 0 (Normal Skin)**:

### Strategy 1: Controlled Normal Skin Volunteer Dataset (Recommended for Hackathon)
- Collect standardized smartphone close-ups of **unaffected, normal skin** (forearms, inner arms, hands, legs) across varied skin tones (Fitzpatrick I–VI / Monk 1–10) under standard ambient and indoor lighting.
- Verified absence of active lesions, rashes, or moles in the target region.

### Strategy 2: Dermoscopic Margin Normal Skin Patches (for Dermoscopy Domain)
- Crop verified non-lesional healthy skin borders surrounding benign lesions from HAM10000/ISIC.
- *Caveat:* Must exclude dermoscopic immersion fluid rings, vignette edges, or hair artifacts.

### Strategy 3: Dedicated Normal Skin Control Benchmarks
- Use established open-source dermatology control cohorts (e.g. non-lesional healthy volunteer skin patches from clinical datasets such as DermNet normal controls or PAD-UFES-20 healthy control patches).

---

## 7. Proposed Normalized Metadata Schema

To ensure seamless integration across datasets without silent relabeling, all images will be indexed using the following standardized schema:

```csv
image_path,dataset_source,original_label,malignancy_status,binary_label,patient_id,skin_tone,image_domain,label_status,exclusion_reason
```

A live preview sample has been generated and saved to [`data_normal_disease/normalized_metadata_sample.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data_normal_disease/normalized_metadata_sample.csv):

| image_path | dataset_source | original_label | malignancy_status | binary_label | patient_id | skin_tone | image_domain | label_status | exclusion_reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `.../000001.png` | Stanford_DDI | melanoma-in-situ | malignant | **UNKNOWN** | UNAVAILABLE | FST_56 | clinical_photography | verified_lesion | Pending binary mapping approval |
| `.../000005.png` | Stanford_DDI | basal-cell-carcinoma | malignant | **UNKNOWN** | UNAVAILABLE | FST_12 | clinical_photography | verified_lesion | Pending binary mapping approval |
| `.../000010.png` | Stanford_DDI | bcc-superficial | malignant | **UNKNOWN** | UNAVAILABLE | FST_34 | clinical_photography | verified_lesion | Pending binary mapping approval |

> `binary_label` remains strictly **UNKNOWN** until the mapping rules and normal control sources are formally approved.

---

## 8. Stage A Sign-off & Stop

- **Experiment Folders Created:**
  - [`data_normal_disease/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data_normal_disease)
  - [`models/normal_disease/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/normal_disease)
  - [`outputs/normal_disease/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/normal_disease)
  - [`reports/normal_disease/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/normal_disease)
- **Code & Model Safety:** Zero existing models (`models/*.keras`), splits (`outputs/*.csv`), or configurations were touched.
- **Stage Status:** **Stage A complete.** No training has been executed. Awaiting your approval before proceeding to Stage B (Label Mapping & Split Plan).
