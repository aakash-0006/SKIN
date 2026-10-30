# Stage B Manifest & Subject-Disjoint Partition Report: Normal Hand Skin Model

**Date:** October 29, 2026  
**Auditor / Engineer:** Senior Deep Learning Engineer & Medical Computer Vision Specialist  
**Reference Audit:** [`reports/experiments/normal_hand_dataset_audit.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/normal_hand_dataset_audit.md)  
**Manifest Path:** [`data/experiments/normal_hand_v1/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv)  
**Manifest SHA-256 Hash:** `5489a71d0888b53dfdf61150984d66a1f3266d868a666ff5692bcd453a402203`  
**Current Milestone:** Stage B — Manifest Generation & Subject-Disjoint Partitioning (Pre-Modeling Checkpoint)  

---

## 1. Executive Summary & Verification

Stage B construction has established an isolated, reproducible, leakage-free experiment manifest for normal-looking hand skin modeling.

All requirements mandated by the Stage B protocol have been satisfied:
1. **Isolated Manifest Creation:** Manifest created at `data/experiments/normal_hand_v1/manifest_v1.csv` without modifying any existing production assets.
2. **Exact Duplicate Exclusion:** Identified and excluded `Hand_0009559.jpg` (byte-for-byte exact MD5 duplicate of `Hand_0009558.jpg`, MD5: `1f2c97dee29d90dff358c1d0ffd6d6d8`).
3. **Quarantine of Flagged Participants:** All 4 participants with source visual irregularities (Participant IDs `27`, `570`, `1566`, and `1000000`, totaling 243 images) have been completely segregated into a standalone `exploratory_evaluation` partition. They are excluded from baseline training and validation.
4. **Subject-Disjoint Splitting:** The remaining 185 eligible participants are split into approximately 70% Train (129 participants), 15% Validation (28 participants), and 15% Test (28 participants).
5. **Zero Data Leakage:** All images from any given participant reside strictly in a single partition. Participant overlap across all partitions is exactly zero.
6. **Integrity Preservation:** Original metadata columns and labels (`irregularities`, `accessories`, `nail_polish`, `aspectOfHand`, `skinColor`, `gender`, `age`) are preserved exactly. No medical diagnoses were assigned or assumed.

---

## 2. Dataset Sources and Inventory Reconciliations

* **Source Metadata:** `C:\Users\USER\Downloads\normal skin dataset\archive (3)\HandInfo.csv`
* **Source Image Directory:** `C:\Users\USER\Downloads\normal skin dataset\archive (3)\Hands\Hands`
* **Total Records in Source:** **`11,076`**
* **Total Images on Disk:** **`11,076`** (100% verified present and readable)
* **Total Records in Manifest:** **`11,076`** (Reconciled with zero missing or extra records)

---

## 3. Partition Summary & Counts

| Partition | Participant Count | % of Eligible Participants | Image Count | % of Eligible Images | Accessories Present | Nail Polish Present | Irregularities Flagged |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **129** | 69.73% | **7,655** | 70.67% | 2,113 (27.6%) | 751 (9.8%) | 0 (0.0%) |
| **`val`** | **28** | 15.14% | **1,538** | 14.20% | 378 (24.6%) | 174 (11.3%) | 0 (0.0%) |
| **`test`** | **28** | 15.14% | **1,639** | 15.13% | 720 (43.9%) | 142 (8.7%) | 0 (0.0%) |
| **`exploratory_evaluation`** (Quarantined) | **4** | N/A (Quarantined) | **243** | N/A (Quarantined) | 0 (0.0%) | 0 (0.0%) | 164 (67.5%)* |
| **`excluded`** (Duplicate) | **1** | N/A (Excluded) | **1** | N/A (Excluded) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Total Manifest Records** | **189** | **100.0%** | **11,076** | **100.0%** | **3,211** | **1,067** | **164** |

*\*Note on Exploratory Evaluation:* The 4 flagged participants contributed 243 total images: 164 were flagged with `irregularities = 1` in the source dataset, and 79 had `irregularities = 0` (e.g., contralateral hand or alternative pose). Because participant separation requires keeping all images of an individual together, all 243 images are quarantined into `exploratory_evaluation`.

---

## 4. Metadata Distributions Across Partitions

### A. Anatomical Aspect Distribution
All 4 primary hand aspects (dorsal right, palmar right, dorsal left, palmar left) maintain balanced representation across partitions:

| Partition | Dorsal Left | Dorsal Right | Palmar Left | Palmar Right | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | 1,939 (25.3%) | 1,972 (25.8%) | 1,770 (23.1%) | 1,974 (25.8%) | 7,655 |
| `val` | 382 (24.8%) | 420 (27.3%) | 358 (23.3%) | 378 (24.6%) | 1,538 |
| `test` | 427 (26.1%) | 431 (26.3%) | 397 (24.2%) | 384 (23.4%) | 1,639 |
| `exploratory_evaluation` | 40 (16.5%) | 69 (28.4%) | 58 (23.9%) | 76 (31.3%) | 243 |
| `excluded` | 0 | 1 | 0 | 0 | 1 |
| **Total** | **2,788** | **2,893** | **2,583** | **2,812** | **11,076** |

### B. Skin Tone Annotations Across Partitions

| Partition | Fair | Medium | Dark | Very Fair | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | 2,347 (30.7%) | 4,539 (59.3%) | 557 (7.3%) | 212 (2.8%) | 7,655 |
| `val` | 511 (33.2%) | 903 (58.7%) | 55 (3.6%) | 69 (4.5%) | 1,538 |
| `test` | 595 (36.3%) | 993 (60.6%) | 51 (3.1%) | 0 (0.0%) | 1,639 |
| `exploratory_evaluation` | 39 (16.0%) | 120 (49.4%) | 84 (34.6%) | 0 (0.0%) | 243 |
| `excluded` | 1 | 0 | 0 | 0 | 1 |
| **Total** | **3,493** | **6,555** | **747** | **281** | **11,076** |

### C. Gender Distribution Across Partitions
Stratified splitting by participant gender preserved demographic balance:
* `train`: 4,892 Female (63.9%), 2,763 Male (36.1%)
* `val`: 1,027 Female (66.8%), 511 Male (33.2%)
* `test`: 1,189 Female (72.5%), 450 Male (27.5%)
* `exploratory_evaluation`: 21 Female (8.6%), 222 Male (91.4%)
* `excluded`: 1 Female (100.0%)

---

## 5. Leakage Verification & Validation Suite

Automated assertions executed against the manifest confirmed all safety constraints:

1. **Physical Path Existence:** `11,076 / 11,076` file paths exist on the local filesystem and are non-empty.
2. **Subject Disjointness:** Every participant is assigned to exactly **one** partition.
3. **Partition Non-Overlap:**
   * $\text{Train} \cap \text{Val} = \emptyset$ (0 participants overlap)
   * $\text{Train} \cap \text{Test} = \emptyset$ (0 participants overlap)
   * $\text{Val} \cap \text{Test} = \emptyset$ (0 participants overlap)
   * $\text{Baseline} \cap \text{Exploratory} = \emptyset$ (0 participants overlap)
4. **Duplicate Exclusion Documented:** Record `Hand_0009559.jpg` is isolated with `status = "excluded"` and detailed justification recorded.
5. **Irregularities Quarantined:** 100% of images from Subjects `27`, `570`, `1566`, and `1000000` reside strictly in `exploratory_evaluation`. Baseline training and validation contain exactly **zero** images from these subjects.
6. **No Medical Relabeling:** Ground truth values `irregularities \in {0, 1}` were preserved verbatim; no clinical disease diagnoses were assumed or created.
7. **Documented Reasons:** Every excluded and quarantined record possesses an explicit explanatory string.
8. **Total Accounting:** $7,655 + 1,538 + 1,639 + 243 + 1 = 11,076$ rows.

---

## 6. Random Seed & Methodology Documentation

* **Partition Strategy:** Subject-Level Stratified Partitioning.
* **Eligible Cohort:** 185 unique participants (excluding the 4 quarantined subjects).
* **Random Seed:** `42` (deterministic `sklearn.model_selection.train_test_split`).
* **Stratification Target:** Participant gender (`eligible_participants["gender"]`).
* **Two-Stage Split:**
  * Stage 1: 70% Train ($N=129$), 30% Temporary ($N=56$).
  * Stage 2: 50% of Temporary to Validation ($N=28$), 50% to Test ($N=28$).

---

## 7. Warnings and Unresolved Limitations

1. **Young Adult Age Skew:** The average age in the baseline cohort is 22.1 years (median 21.0). Hands belonging to older adults with prominent veins, solar lentigines, or reduced turgor are underrepresented in baseline training.
2. **Flat White Studio Background:** All images in 11k Hands share a flat white paper background. Models trained on uncropped images will learn to rely on the white background. Hand bounding-box cropping remains mandatory during future modeling stages.
3. **Exploratory Evaluation Role:** The `exploratory_evaluation` partition is reserved strictly for out-of-distribution / anomaly testing after baseline training is finalized. It must never be used for tuning hyperparameters or loss weights.

---

## 8. Checkpoint Status

**Stage B is complete.**
* The manifest is written, verified, and sealed.
* **Manifest Path:** `D:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI\data\experiments\normal_hand_v1\manifest_v1.csv`
* **Next Milestone:** Stage C — Confirm that the available labels support the selected learning task.
* **Action:** STOPPING and awaiting user approval before proceeding.
