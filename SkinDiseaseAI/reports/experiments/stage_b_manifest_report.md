# 📋 SkinDiseaseAI — Stage B: Isolated Experiment Manifest Construction & Integrity Report

**Project:** SkinDiseaseAI — Full Dataset Integration, Augmentation, Retraining and Accuracy Improvement  
**Author:** Senior Machine Learning Engineer, Deep Learning Researcher & Medical CV Specialist  
**Date:** October 29, 2026  
**Document:** `reports/experiments/stage_b_manifest_report.md`  
**Execution Environment:** Windows 11 | Python 3.12.10 | TensorFlow 2.21.0 | 24 GB RAM | NVIDIA RTX 3050  
**Status:** Stage B Complete — Manifest Generated, Validated, and Stopped for User Review  

---

## 1. Executive Summary & Generated Artifacts

In strict compliance with the **Stage B** requirements, an isolated, grouped experiment manifest has been constructed linking clean, non-overlapping records from **HAM10000** and **ISIC 2019**. The dataset creation process was performed with zero file modification to protected production assets and zero external dataset downloads.

### Primary Stage B Artifacts Generated:
1. **Experiment Manifest:** [`data/experiments/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/manifest_v1.csv)
   - **Total Records:** **23,463 images** (`train`: 18,529 | `val`: 3,440 | `test`: 1,494)
   - **File Size:** 3,483,870 bytes
   - **SHA-256 Digest:** `27750dd6d99352b93d9ad805bf94ad875e492b014112f70855ba27c91247bc4a`
2. **Traceable Exclusions Log:** [`data/experiments/exclusions_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/exclusions_v1.csv)
   - **Total Excluded Records:** **11,883 images** (100% traceable with specific category & rationale)
   - **File Size:** 1,268,590 bytes
   - **SHA-256 Digest:** `beef2fe7bcc0f110c0ec2ee74c52d3d77d07d0d85fa38a4faf8a3512d9ae9c14`

### Key Verification Highlights:
- **100% Preserved Frozen Test Set:** The 1,494 frozen HAM10000 test images remain byte-for-byte identical to `outputs/splits.csv`. Zero test images appear in training or validation.
- **Lesion-Level Integrity:** Zero known lesion groups cross train, validation, or test boundaries.
- **Unlinked ISIC Records Handled Safely:** The 1,983 clean ISIC images lacking source `lesion_id` are assigned unique image-level groups (`group_id = image:<image_id>`) and allocated strictly to the training fold, ensuring zero unlinked patient leakage into the validation benchmark.
- **Stanford DDI Fully Isolated:** All 656 clinical photographs in Stanford DDI remain 100% external. Zero DDI images appear in the manifest.
- **Physical Image Existence:** Exactly 23,463 of 23,463 image file paths (100.0%) were programmatically verified to exist on disk.
- **Protected Baseline Integrity:** All 8 protected project files and the master archive `SkinDiseaseAI_v1_20261027.zip` remain 100% intact with matching SHA-256 digests.

---

## 2. Candidate Source Counts & Reconciliation

An exhaustive audit of all available physical records was executed across the three datasets:

| Dataset | Total On Disk | Included in Manifest | Excluded Records | Role in Experiment Pipeline |
| :--- | :---: | :---: | :---: | :--- |
| **HAM10000** | 10,015 | **9,934** | 81 | Primary dermoscopy baseline (Train: 6,930, Val: 1,510, Test: 1,494) |
| **ISIC 2019** | 25,331 | **13,529** | 11,802 | Supplementary dermoscopy (Train: 11,599, Val: 1,930, Test: 0) |
| **Stanford DDI** | 656 | **0** | 656 | External evaluation benchmark only (Zero training/validation rows) |
| **Total** | **36,002** | **23,463** | **12,538** | **Full Traceability Reconciled** |

*Note: Total candidate records audited across HAM10000 + ISIC 2019 = $10,015 + 25,331 = 35,346$. Exactly $23,463 \text{ (included)} + 11,883 \text{ (excluded)} = 35,346$.*

---

## 3. Label Taxonomy Standardization & Exclusions

The experiment manifest strictly admits the seven supported dermoscopic classes:
1. `akiec` — Actinic keratoses and intraepithelial carcinoma / Bowen's disease
2. `bcc` — Basal cell carcinoma
3. `bkl` — Benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
4. `df` — Dermatofibroma
5. `mel` — Melanoma
6. `nv` — Melanocytic nevus
7. `vasc` — Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

### Diagnostic Incompatibility Exclusions:
- **ISIC 2019 `SCC` (Squamous Cell Carcinoma, 628 images):**  
  Invasive squamous cell carcinoma is an independent, aggressive malignancy not represented in the 7-class HAM10000 taxonomy (which only includes pre-cancerous actinic keratoses and intraepithelial Bowen's disease under `akiec`). Mapping `SCC` to `akiec` or any other category would corrupt clinical decision boundaries.  
  **Action:** All 628 `SCC` images are **quarantined and excluded** in `data/experiments/exclusions_v1.csv` with category `incompatible_class_scc`.
- **Stanford DDI (656 images across 78 diagnostic categories):**  
  DDI spans conditions such as epidermal cysts, mycosis fungoides, kaposi sarcoma, and verruca vulgaris. Furthermore, it contains **0 normal skin controls** and was photographed with non-dermoscopic clinical cameras.  
  **Action:** DDI is **completely excluded** from `manifest_v1.csv` and reserved exclusively for post-training out-of-domain external evaluation.

---

## 4. Comprehensive Deduplication & Exclusion Accounting

Deduplication was conducted using exact SHA-256 hashing and perceptual hashing (pHash 64-bit DCT with Hamming distance threshold $\le 4$). Every exclusion was tagged with an explicit category and human-readable explanation in `data/experiments/exclusions_v1.csv`:

### Exclusions Breakdown by Category ($N = 11,883$):

```
┌──────────────────────────────────────┬─────────────┬─────────────┬────────────────────────────────────────────────────────┐
│ Exclusion Category                   │ Source Data │ Count       │ Rationale & Safety Impact                              │
├──────────────────────────────────────┼─────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ near_ham_duplicate                   │ ISIC 2019   │ 9,365       │ Near-duplicate match with HAM10000 (Hamming <= 4)      │
│ internal_isic_near_duplicate         │ ISIC 2019   │ 1,809       │ Near-duplicate pairs within ISIC 2019 archive          │
│ incompatible_class_scc               │ ISIC 2019   │ 628         │ Invasive SCC outside target 7-class label taxonomy     │
│ cross_split_val_conflict             │ HAM10000    │ 46          │ HAM training image near-duplicate of HAM val image     │
│ cross_split_test_conflict            │ HAM10000    │ 35          │ HAM train/val image near-duplicate of frozen test image│
├──────────────────────────────────────┼─────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ Total Excluded Records               │             │ 11,883      │ 100% Traceable in data/experiments/exclusions_v1.csv   │
└──────────────────────────────────────┴─────────────┴─────────────┴────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Test Set Protection:** Exactly 35 HAM10000 training/validation images and 3,141 candidate ISIC images that exhibited exact or near-duplicate similarity with the 1,494 frozen HAM test images were pruned from the training and validation pipelines. This guarantees that the test set remains completely uncontaminated by identical or near-identical lesions.

---

## 5. Identifier Availability & Grouping Analysis

### Lesion Tracking Summary:
- **HAM10000 ($N = 9,934$ included):**
  - Unique `lesion_id`s: **7,470**
  - Every image has a valid, documented `lesion_id`.
  - Group identifier format: `lesion:HAM_xxxxxxx`.
- **ISIC 2019 ($N = 13,529$ included):**
  - **With valid `lesion_id` ($N = 11,546$):** Group identifier format: `lesion:MSK4_xxxxxxx` or `lesion:BCN_xxxxxxx`. All images sharing the same `lesion_id` are grouped together.
  - **Without source `lesion_id` ($N = 1,983$):** Sourced from hospital archives (primarily BCN20000) where clinical anonymization removed lesion linkage.
    - Assigned unique image-level groups: `group_id = image:<image_id>`.
    - `has_lesion_id`: `False`.
    - `is_unlinked_single_sample`: `True`.

### Allocation of Unlinked Records & Leakage Warning:
To resolve Requirement 4 and adhere to the Stage A audit recommendation:
1. **Strict Placement in Training Fold:** All 1,983 unlinked clean ISIC records were allocated strictly to the `train` partition.
2. **Protection of the Validation Set:** Because zero unlinked records are placed in `val`, the validation set ($N = 3,440$) consists exclusively of records with verified, non-overlapping `lesion_id`s. This completely eliminates any possibility of unlinked patient cross-leakage between training and validation.
3. **Explicit Leakage Warning:**  
   > [!WARNING]  
   > **Unlinked Lesion Leakage Limitation:** For the 1,983 ISIC records lacking source `lesion_id`, **internal lesion-level and patient-level clustering within the training fold cannot be ruled out**. If an unlinked patient in BCN20000 had multiple lesions photographed, those lesions exist as independent single-sample records in the training set. However, because none of these records exist in `val` or `test`, **they cannot cause optimistic evaluation leakage**.

---

## 6. Manifest Composition & Split Distribution

The generated manifest [`data/experiments/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/manifest_v1.csv) contains the following finalized partitions:

### Partition Breakdown by Source Dataset:

| Dataset Source | Train (`train`) | Validation (`val`) | Frozen Test (`test`) | Total Included |
| :--- | :---: | :---: | :---: | :---: |
| **HAM10000** | 6,930 | 1,510 | 1,494 | **9,934** |
| **ISIC 2019 (Linked)** | 9,616 | 1,930 | 0 | **11,546** |
| **ISIC 2019 (Unlinked Single-Sample)**| 1,983 | 0 | 0 | **1,983** |
| **Total Combined** | **18,529** | **3,440** | **1,494** | **23,463** |

### Class Distribution Across Partitions:

| Class Label | Clinical Category | Train ($N=18,529$) | Val ($N=3,440$) | Test ($N=1,494$) | Total ($N=23,463$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `akiec` | Actinic keratosis | 690 | 208 | 63 | **961** |
| `bcc` | Basal cell carcinoma | 2,396 | 523 | 68 | **2,987** |
| `bkl` | Benign keratosis | 1,944 | 393 | 152 | **2,489** |
| `df` | Dermatofibroma | 173 | 44 | 7 | **224** |
| `mel` | Malignant melanoma | 3,454 | 629 | 187 | **4,270** |
| `nv` | Melanocytic nevus | 9,698 | 1,604 | 996 | **12,298** |
| `vasc` | Vascular lesion | 174 | 39 | 21 | **234** |
| **Total** | | **18,529** | **3,440** | **1,494** | **23,463** |

### Unlinked Status by Partition:

| Partition | Linked Lesions (`has_lesion_id == True`) | Unlinked Single-Sample (`has_lesion_id == False`) | Total Images |
| :--- | :---: | :---: | :---: |
| **Train** | 16,546 (89.3%) | 1,983 (10.7%) | 18,529 |
| **Validation**| 3,440 (100.0%) | 0 (0.0%) | 3,440 |
| **Frozen Test**| 1,494 (100.0%) | 0 (0.0%) | 1,494 |

---

## 7. Strict Integrity Assertions & Verification Results

The assertion verification script was executed directly on `manifest_v1.csv` and `exclusions_v1.csv`. All six mandatory conditions passed without error:

```
================ RUNNING STRICT STAGE B ASSERTIONS ================
[PASS] Assertion 1: No frozen HAM10000 test image appears in training or validation.
       - Frozen test set size: exactly 1,494 images.
       - Test image overlap with train/val: 0 images (0.0%).
       - Exact match between manifest test split and outputs/splits.csv: TRUE.

[PASS] Assertion 2: No known lesion group crosses train/validation/test.
       - Train vs. Val group overlap: 0 groups.
       - Train vs. Test group overlap: 0 groups.
       - Val vs. Test group overlap: 0 groups.
       - Genuine lesion ID overlap across all splits: 0 lesions.

[PASS] Assertion 3: No Stanford DDI image appears in the training or validation manifest.
       - Stanford DDI overlap with manifest: 0 images (0.0%).
       - Stanford DDI records in training/val: NONE (100% external benchmark).

[PASS] Assertion 4: No unsupported class appears in the manifest.
       - Unique manifest classes: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'].
       - Unsupported classes (SCC, UNK, invalid): 0 records.
       - Integer targets y: all values strictly in range [0, 6].

[PASS] Assertion 5: Every included image path exists.
       - Total paths evaluated: 23,463.
       - Missing or unreadable files: 0.
       - Physical existence rate: 100.0%.

[PASS] Assertion 6: All exclusions are traceable.
       - Total candidate records evaluated: 35,346.
       - Total exclusions recorded: 11,883.
       - Reconciliation: 23,463 (included) + 11,883 (excluded) == 35,346.
       - Unattributed exclusions: 0 (all records possess category & rationale).
================ ALL STAGE B ASSERTIONS PASSED ================
```

---

## 8. Protected Asset Verification & Hash Table

To guarantee zero regression or accidental corruption of baseline assets, all protected files and the master ZIP archive were verified via SHA-256:

| Protected File Path | Physical Existence | Verified Size | SHA-256 Digest | Verification Status |
| :--- | :--- | :---: | :--- | :--- |
| [`models/best_model.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/best_model.keras) | **EXISTS** | 33,486,076 B | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **Intact / Unchanged** |
| [`models/isic_finetuned.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_finetuned.keras) | **EXISTS** | 33,477,534 B | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **Intact / Unchanged** |
| [`models/isic_augmented_v1.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_augmented_v1.keras) | **EXISTS** | 33,477,861 B | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **Intact / Unchanged** |
| [`models/ood_gate.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/ood_gate.npz) | **EXISTS** | 6,102,303 B | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **Intact / Unchanged** |
| [`outputs/splits.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/splits.csv) | **EXISTS** | 1,259,807 B | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **Intact / Unchanged** |
| [`outputs/combined_splits.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/combined_splits.csv) | **EXISTS** | 2,556,961 B | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **Intact / Unchanged** |
| [`config.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/config.json) | **EXISTS** | 1,985 B | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **Intact / Unchanged** |
| [`app.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/app.py) | **EXISTS** | 16,223 B | `8ed50cddfde86d99723ec09bf1fbf59a16f849e7bdf8f2bb7555e1fc45f28b74` | **Intact / Unchanged** |
| [`SkinDiseaseAI_v1_20261027.zip`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI_v1_20261027.zip) | **EXISTS** | 1,506,237,677 B | `11badbdd6b32140e55660c7ea2d3a499a3ba197ea222d29cb06b5bdaabe9bacd` | **Locked Master Backup** |

---

## 9. Stage C Preparation Roadmap & Verification Checkpoints

With the isolated manifest successfully constructed, the project is ready for **Stage C: Position-Aware Image Augmentation**.

### Proposed Stage C Implementation Principles:
1. **On-The-Fly Processing Only:** Augmentation will be implemented as a Keras/TensorFlow preprocessing pipeline active strictly when `training=True`. **Zero offline variant files will be generated or saved to disk.**
2. **Position-Aware Translation Framing:**
   - Sample horizontal shift parameters directing the lesion center toward left ($\Delta x \in [-0.15, -0.05]$), center ($\Delta x \in [-0.05, +0.05]$), and right ($\Delta x \in [+0.05, +0.15]$) of the $224 \times 224$ frame.
   - Restrict cropping bounds to ensure $\ge 70\%$ of usable lesion area and diagnostic boundary contours remain inside the optical field.
3. **Conservative Lighting & Geometric Jitter:**
   - Rotation: $\pm 15^\circ$ maximum (preventing orientation distortion).
   - Horizontal flip only (vertical flips strictly prohibited for dermatological asymmetry validity).
   - Brightness: $\pm 8\%$ jitter; Contrast: $\pm 8\%$ jitter.
4. **Strict Evaluation Isolation:**
   - Augmentation pipeline will be 100% bypassed (`training=False`) during validation and test evaluations.

---

## 10. Stage B Sign-Off & Strict Stop

- **Manifest Created:** [`data/experiments/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/manifest_v1.csv) (23,463 records, SHA-256: `27750dd6d993...`)
- **Exclusions Logged:** [`data/experiments/exclusions_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/exclusions_v1.csv) (11,883 records, SHA-256: `beef2fe7bcc0...`)
- **Assertions Passed:** All 6 integrity checks verified programmatically.
- **Safety Compliance:** Zero models retrained, zero files downloaded, zero protected assets modified.
- **Report Saved:** Saved to [`reports/experiments/stage_b_manifest_report.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/stage_b_manifest_report.md).

🛑 **STAGE B COMPLETE — STOPPING FOR YOUR REVIEW AND APPROVAL.**  
Please review the generated manifest, exclusions log, and verification results above. Let me know when you approve proceeding to **Stage C (Position-aware image augmentation design and testing)**.
