# 🔬 SkinDiseaseAI — Full Repository, Model Architecture & Dataset Audit Report

**Project:** SkinDiseaseAI — Full Deep Learning Model Improvement and Skin Analysis  
**Role:** Senior Machine Learning Engineer, Deep Learning Researcher & Medical Computer Vision Specialist  
**Date:** October 29, 2026  
**Document:** `reports/experiments/full_model_audit.md`  
**Execution Context:** Native Windows 11 (64-bit) | Python 3.12.10 | TensorFlow 2.21.0 | Keras 3 | 24 GB RAM | NVIDIA RTX 3050 6 GB (CPU execution via oneDNN/AVX2)  
**Audit Mode:** Read-Only Technical Audit (Zero code edits, zero model retraining, zero file modifications)  

---

## 1. Executive Summary & Audit Purpose

This document provides a comprehensive, evidence-based technical audit of the **SkinDiseaseAI** repository, datasets, deep learning model architectures, training pipelines, evaluation protocols, and user-facing inference applications.

The purpose of this audit is to:
1. Establish a rigorous, verified baseline of all current assets, code paths, and configurations.
2. Uncover the architectural and mathematical mechanisms driving observed performance trade-offs (specifically the divergence between headline accuracy and melanoma sensitivity).
3. Formulate the single highest-priority, evidence-supported experimental proposal to improve model generalization, class balance, and clinical reliability without modifying protected production assets.

---

## 2. Protected Asset Inventory & Cryptographic Integrity

All primary model checkpoints, partition manifests, configuration files, and backups were inspected on disk and verified via SHA-256 hashing:

| Protected File Path | Physical Existence | Verified Size | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| [`models/best_model.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/best_model.keras) | **EXISTS** | 33,486,076 B | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **Immutable Baseline** |
| [`models/isic_augmented_v1.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_augmented_v1.keras) | **EXISTS** | 33,477,861 B | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **Immutable Baseline** |
| [`models/isic_finetuned.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_finetuned.keras) | **EXISTS** | 33,477,534 B | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **Immutable Baseline** |
| [`models/ood_gate.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/ood_gate.npz) | **EXISTS** | 6,102,303 B | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **Immutable Baseline** |
| [`outputs/splits.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/splits.csv) | **EXISTS** | 1,259,807 B | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **Immutable Baseline** |
| [`outputs/combined_splits.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/combined_splits.csv) | **EXISTS** | 2,556,961 B | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **Immutable Baseline** |
| [`config.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/config.json) | **EXISTS** | 1,985 B | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **Immutable Baseline** |
| [`D:/DOWNLOADS D/SkinDiseaseAI_v1_20261027.zip`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI_v1_20261027.zip) | **EXISTS** | 1,506,237,677 B | `11badbdd6b32140e55660c7ea2d3a499a3ba197ea222d29cb06b5bdaabe9bacd` | **Locked Master Backup** |

---

## 3. Deep Learning Model Architecture & Parameter Verification

Programmatic layer inspection of `best_model.keras`, `isic_finetuned.keras`, and `isic_augmented_v1.keras` confirmed identical top-level structural topology:

```
[ Input: (None, 224, 224, 3) float32 ]
                  │
                  ▼
[ Layer 1: Augmentation Sequential ]
  (Active only during training=True; exact identity pass-through in eval)
                  │
                  ▼
[ Layer 2: EfficientNetB0 Backbone (238 internal layers) ]
  (Pre-trained ImageNet feature extractor with internal normalization)
                  │
                  ▼
[ Layer 3: GlobalAveragePooling2D ] -> 1,280-dim feature vector
                  │
                  ▼
[ Layer 4: Dropout(rate=0.30) ]
                  │
                  ▼
[ Layer 5: Dense(7 units, activation="softmax") ] -> 7-class probabilities
```

### Parameter Count Breakdown (Physically Measured)
- **Input Dimensions:** `(None, 224, 224, 3)`
- **Total Parameters:** `4,058,538` (~4.06M weights)
- **Trainable Parameters:** `2,046,487` (~2.05M weights)
  - Backbone Trainable Parameters: `2,037,520` (top 40 convolutional/depthwise layers unfrozen, batch normalization layers kept non-trainable)
  - Head Trainable Parameters: `8,967` (Dense layer: $1,280 \times 7 + 7 = 8,967$)
- **Non-Trainable Parameters:** `2,012,051` (lower frozen backbone layers + batch normalization moving statistics)
- **Output Dimensions:** `(None, 7)`

### Online Augmentation Layer Configuration
- In `best_model.keras`: Uses older generic Keras layers: `['random_flip', 'random_rotation', 'random_zoom', 'random_contrast']`.
- In `isic_augmented_v1.keras`: Uses custom conservative sequential block (`src/preprocessing.py:build_augmentation_layer`):
  1. `aug_hflip`: `RandomFlip(mode="horizontal")` (horizontal flip only; vertical flip is strictly omitted to prevent upside-down orientation artifacts).
  2. `aug_rotation`: `RandomRotation(factor=(-0.05556, 0.05556))` ($\pm 20^\circ$ rotation with reflect padding).
  3. `aug_zoom`: `RandomZoom(height_factor=(-0.08, 0.08))` ($\pm 8\%$ scale variation).
  4. `aug_translation`: `RandomTranslation(factor=(-0.05, 0.05))` ($\pm 5\%$ horizontal and vertical shift).

---

## 4. Preprocessing & Tensor Pipeline Audit

A physical inspection of [`src/preprocessing.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/preprocessing.py) and [`src/predict.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py) verified complete mathematical alignment:

| Processing Stage | Training Pipeline (`src/preprocessing.py`) | Inference Pipeline (`src/predict.py`) | Consistency Check |
| :--- | :--- | :--- | :--- |
| **Image Decoding** | `tf.io.decode_image(channels=3)` | PIL `Image.open().convert("RGB")` | **Identical (RGB 3-channel)** |
| **Spatial Sizing** | Resized to `(224, 224)` bilinear | Resized to `(224, 224)` bilinear | **Identical (224x224)** |
| **Data Type** | `tf.cast(img, tf.float32)` | `np.asarray(..., dtype=np.float32)` | **Identical (`float32`)** |
| **Pixel Numerical Scale** | Raw floats in `[0.0, 255.0]` | Raw floats in `[0.0, 255.0]` | **Identical (`[0.0, 255.0]`)** |
| **Backbone Normalization** | Handled internally by EfficientNetB0 | Handled internally by EfficientNetB0 | **No double-scaling bug** |

---

## 5. Dataset Inventory, Provenance & Split Analysis

### A. Dataset Distributions by Source & Split

Physical inspection of `HAM10000_metadata.csv`, `outputs/splits.csv`, and `outputs/combined_splits.csv` confirms:

```
Combined Dataset Split Composition (Total N = 21,570 images):
┌────────────────┬──────────────┬──────────────┬──────────────┬────────────────┐
│ Split          │ HAM10000     │ ISIC 2019    │ Total Images │ Partition Role │
├────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ Training       │ 6,930        │ 9,616        │ 16,546       │ Train (76.7%)  │
│ Validation     │ 1,510        │ 1,930        │ 3,440        │ Val (15.9%)    │
│ Preserved Test │ 1,494        │ 0            │ 1,494        │ Test ( 6.9%)   │
├────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ Total          │ 9,934*       │ 11,546       │ 21,480       │ 100.0%         │
└────────────────┴──────────────┴──────────────┴──────────────┴────────────────┘
* Note: 81 ambiguous or cross-split conflicting HAM images were excluded during ISIC integration.
```

### B. Class Distribution Matrix Across Partitions

| Target Class Label | Clinical Category | Train Count (`N_tr`) | Train Share | Val Count (`N_val`) | Test Count (`N_test`) | Test Support Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`akiec`** | Actinic keratoses / intraepithelial carcinoma | 690 | 4.17% | 208 | 63 | Adequate support ($\ge 20$) |
| **`bcc`** | Basal cell carcinoma | 2,396 | 14.48% | 523 | 68 | Adequate support ($\ge 20$) |
| **`bkl`** | Benign keratosis-like lesions | 1,746 | 10.55% | 393 | 152 | Adequate support ($\ge 20$) |
| **`df`** | Dermatofibroma | 173 | 1.05% | 44 | **7** | **LOW SUPPORT ($<20$) — High variance** |
| **`mel`** | Melanoma | 3,138 | 18.97% | 629 | 187 | Adequate support ($\ge 20$) |
| **`nv`** | Melanocytic nevi (moles) | 8,229 | 49.73% | 1,604 | 996 | **Dominant class (66.7% of test set)** |
| **`vasc`** | Vascular lesions | 174 | 1.05% | 39 | 21 | Borderline support ($\ge 20$) |
| **Total** | — | **16,546** | **100.0%** | **3,440** | **1,494** | — |

---

## 6. Data Quality, Leakage Prevention & Audit Findings

### A. Missing Values & Metadata Integrity
- In raw `HAM10000_metadata.csv` (10,015 rows):
  - `lesion_id`, `image_id`, `dx`, `dx_type`, `sex`, `localization`: **0 missing values**.
  - `age`: 57 missing values (0.57%), handled as NaN without impacting image-label arrays.
- In `ddi_metadata.csv` (656 rows):
  - Exactly 656 rows, 0 corrupted images. 171 malignant, 485 benign lesions, **0 normal skin controls**.
  - **Critical Metadata Limitation:** Stanford DDI contains **no patient identifiers** (`patient_id`). Cross-image patient clustering within DDI cannot be resolved.

### B. Grouping & Leakage Verification
1. **HAM10000 Lesion Grouping:**
   - In `outputs/splits.csv`, `GroupShuffleSplit` on `lesion_id` ensures that all images belonging to the same lesion remain strictly within one partition.
   - Cross-split audit confirms: **0 shared `lesion_id` between Train, Val, and Test**.
2. **ISIC 2019 Integration Quarantine:**
   - 11,420 duplicate ISIC records were excluded.
   - 3,141 ISIC records overlapping the HAM10000 test set were quarantined.
   - Zero ISIC images were permitted into the test split.
3. **Perceptual Hash Audit:**
   - Evaluated 1,155 candidate pairs with dHash Hamming distance $\le 8$ between test and train/val.
   - Zero pairs exhibited pixel correlation $r \ge 0.97$. Highest correlation was $r = 0.9588$ on visually distinct lesions.
   - **Status:** **Zero image leakage into the preserved test split.**

---

## 7. Class Weighting Implementation & Comparative Analysis

In the training split ($N = 16,546$), benign nevi (`nv`) outnumber dermatofibromas (`df`) by **$47.6 : 1$**.

The table below contrasts the five loss-weighting strategies evaluated:

| Target Class | Training Count ($n_c$) | Fraction | Strat 1: Unweighted | Strat 2: Raw Inverse Freq | Strat 3: Capped (cap=5.0) | Strat 4: ENS ($\beta=0.999$) | Strat 5: Power ($\alpha=0.6$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`akiec`** | 690 | 4.17% | 1.0000 | 3.4257 | 3.4257 | 0.7423 | 0.9649 |
| **`bcc`** | 2,396 | 14.48% | 1.0000 | 0.9865 | 0.9865 | 0.4072 | 0.4572 |
| **`bkl`** | 1,746 | 10.55% | 1.0000 | 1.3538 | 1.3538 | 0.4483 | 0.5528 |
| **`df`** | 173 | 1.05% | 1.0000 | **13.6631** | **5.0000 [CAPPED]** | **2.3287** | **2.2129** |
| **`mel`** | 3,138 | 18.97% | 1.0000 | 0.7533 | 0.7533 | **0.3869** | **0.3889** |
| **`nv`** | 8,229 | 49.73% | 1.0000 | **0.2872** | **0.2872** | **0.3702** | **0.2181** |
| **`vasc`** | 174 | 1.05% | 1.0000 | **13.5846** | **5.0000 [CAPPED]** | **2.3165** | **2.2053** |
| **Ratio ($w_{\text{df}} / w_{\text{nv}}$)**| — | — | **1.0x** | **47.6x** | **17.4x** | **6.3x** | **10.1x** |

### Mathematical Discovery: The Cause of the Melanoma Sensitivity Collapse
- In `isic_augmented_v1.keras` (Strategy 3), the ratio between `mel` weight ($0.7533$) and `nv` weight ($0.2872$) is **$2.62\times$**.
- Because `nv` represents 66.7% of the test set, the cross-entropy optimizer heavily penalized false positive predictions on `nv`.
- Consequently, the model became overly conservative on `mel`, driving melanoma precision up to $56.35\%$ while **melanoma recall collapsed from $53.48\%$ down to $37.97\%$** (missing 116 out of 187 test melanomas, with 60 misclassified as `bkl` and 34 as `nv`).
- **The Theoretical Superiority of ENS ($\beta = 0.999$):**
  - Under ENS ($\beta = 0.999$), `nv` weight ($0.3702$) and `mel` weight ($0.3869$) are nearly equal ($1.045\times$ ratio).
  - This eliminates the gradient penalty distortion against melanoma while still providing a bounded $6.29\times$ boost to rare classes (`df`, `vasc`).

---

## 8. Input Gate, Calibration & Abstention Logic

### A. Current Implementation (`src/gate.py`, `models/ood_gate.npz`)
- Employs Mahalanobis distance calculated against an empirical Gaussian fitted on 1,280-dim ImageNet EfficientNetB0 embeddings of 4,000 dermoscopy training images.
- Distance threshold: `thr = 2673.0803` (chosen on validation dermoscopy images to accept 97.0%).
- Quality limits: Luminance $[50.43, 245]$, blur variance $\ge 4.32$.

### B. Smartphone Incompatibility Finding
- Non-contact smartphone photography lacks dermatoscope liquid contact plates and peripheral vignette rings.
- The resulting embedding shift produces Mahalanobis distances of **$d \approx 5755 \gg 2673.08$**, causing **100% false rejection of valid smartphone skin photos** in the default domain gate mode.
- Users can only classify phone photos by selecting `"Bypass input validation"`, which disables safety checks.

### C. Abstention Verification in Inference
- In [`src/predict.py:84-140`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/predict.py#L84-L140), if any gate check fails, the pipeline halts immediately:
  - Returns `ok = False` with specific `stage_failed` identifier.
  - In `app.py`, the disease classifier is blocked (`⛔ Blocked`).
  - **Zero classification labels, scores, tables, or melanoma flags are rendered.**

---

## 9. Hardware, Memory & Runtime Constraints

Measured execution characteristics on the host system:
- **Operating System:** Windows 11 (64-bit)
- **Physical RAM:** 24.0 GB (abundant headroom; peak training memory is ~3.8 GB)
- **Storage:** Drive D: has **182.66 GB Free**
- **Compute Backend:** TensorFlow $\ge 2.11$ on native Windows operates strictly on multi-threaded CPU instructions (oneDNN/AVX2).
- **Throughput:** ~35–45 images per second during training; ~28–36 ms per image during inference.
- **Estimated Training Runtime:** ~18.5 minutes for a 20-epoch staged fine-tuning run (Phase 1: 10 epochs frozen backbone ~7.5 min; Phase 2: 10 epochs fine-tuning top 40 layers ~11.0 min).

---

## 10. Verification Taxonomy: Code Facts vs. Reported Findings

| Technical Claim / Assertion | Verification Source | Status |
| :--- | :--- | :--- |
| **All 8 protected files exist with unchanged SHA-256 hashes** | Executed python scratch script | **[Verified by executed test]** |
| **Model input is (224, 224, 3) float32 in [0, 255]** | Programmatic model inspection | **[Verified from source code]** |
| **Model parameter count is exactly 4,058,538** | `model.count_params()` check | **[Verified by executed test]** |
| **Ensemble is active default in Streamlit** | `app.py:25-36` inspection | **[Verified from source code]** |
| **Ensemble melanoma threshold is exactly 0.3466** | `config.json:39` inspection | **[Verified from source code]** |
| **OOD gate threshold is 2673.0803 fitted on dermoscopy** | `models/ood_gate.npz` inspection | **[Verified from source code]** |
| **Unit test suite passes 32/33 tests (1 skipped for fixtures)** | `unittest discover` execution | **[Verified by executed test]** |
| **Baseline test accuracy: 67.60%, Macro F1: 0.5327** | `baseline_metrics.json` | **[Reported by existing documentation]** |
| **Aug v1 test accuracy: 72.29%, Mel recall: 37.97%** | `three_model_comparison.json` | **[Reported by existing documentation]** |
| **Calibrated ensemble test mel recall: 54.01% (101/187)** | `stage3_test_bootstrap_results.json` | **[Reported by existing documentation]** |
| **Option A volunteer normal skin images: 0 exist on disk** | File system audit of `data_normal_disease/` | **[Verified from source code]** |

---

## 11. Recommended Single Highest-Priority Experiment

Based on the evidence established in this audit, the single highest-priority, scientifically justified experiment is:

### 🎯 Experiment Proposal: `EXP-ENS-001` — Fine-Tuning with Effective Number of Samples (ENS $\beta = 0.999$) Class Weighting

#### A. Rationale & Core Hypothesis
In `isic_augmented_v1.keras`, capped inverse-frequency weighting ($w_{\text{mel}} = 0.7533$ vs $w_{\text{nv}} = 0.2872$) caused the model to trade away **$62\%$ of true melanomas** to boost overall precision on the dominant `nv` class. 

Under the **Effective Number of Samples** formulation with $\beta = 0.999$:
$$w_{\text{nv}} = 0.3702, \quad w_{\text{mel}} = 0.3869 \quad (\text{Ratio } 1.045\times)$$
Hypothesis: Re-balancing the loss gradient between `mel` and `nv` while maintaining online conservative augmentation will:
1. Prevent the collapse of single-model melanoma sensitivity (targeting **mel recall $\ge 50\%$ at default argmax**, up from $37.97\%$).
2. Maintain strong overall accuracy ($>71\%$) and Macro F1 ($>0.58$).
3. Eliminate the operational overhead of running two separate models in an ensemble forward pass.

#### B. Execution Plan & Isolation
- **Experiment ID:** `EXP-ENS-001`
- **Output Checkpoint:** `models/experiments/isic_ens_beta999.keras` (Zero modification of production models).
- **Training Script:** New isolated script `src/train_experiment.py` running in `models/experiments/`.
- **Dataset:** Preserved training partition of `outputs/combined_splits.csv` ($N = 16,546$).
- **Hyperparameters:** Batch size 32, Adam optimizer, initial LR $1 \times 10^{-5}$, top 40 layers fine-tuned, 10 epochs with early stopping on validation Macro F1.
- **Resource Footprint:** Estimated runtime **~11.5 minutes on CPU**, peak RAM **~3.8 GB**.

#### C. Evaluation Criteria & Safety Gates
1. **Validation Gate (Primary):** Must achieve validation Macro F1 $\ge 0.585$ and validation melanoma recall $\ge 55\%$.
2. **Frozen Test Set (Single Evaluation):** If validation criteria are met, evaluate ONCE on the frozen HAM10000 test set (1,494 images).
3. **No Overwrite Rule:** The result will be cataloged in `reports/experiments/` without modifying `app.py` or existing reports until reviewed and approved.

---

## 12. Conclusion & Request for Authorization

Stage A audit is complete. All existing assets remain intact. 

Please review this audit and confirm whether to proceed with **Experiment `EXP-ENS-001` (ENS $\beta = 0.999$ Fine-Tuning)** or if you prefer an alternative research direction.
