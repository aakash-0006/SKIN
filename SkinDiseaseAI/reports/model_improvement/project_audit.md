# 🔍 SkinDiseaseAI — Project Audit Report

**Project:** SkinDiseaseAI — Complete Model Accuracy Improvement and Generalization Project  
**Author:** Senior Machine Learning Engineer & ML Evaluation Specialist  
**Date:** October 29, 2026  
**Document:** `reports/model_improvement/project_audit.md`  
**Execution Context:** Python 3.12.10 | TensorFlow 2.21.0 | Windows 11 (64-bit) | 24 GB RAM | RTX 3050 (CPU execution via oneDNN/AVX2)  

---

## 1. Executive Summary & Audit Scope

This audit provides an exhaustive, evidence-based inspection of the current **SkinDiseaseAI** codebase, datasets, configurations, model architectures, training pipelines, evaluation protocols, and user interfaces before implementing any experimental improvements.

The investigation adheres strictly to the project safety rules:
- **Zero modification of protected assets:** `models/*.keras`, `outputs/splits.csv`, `outputs/combined_splits.csv`, `config.json`, and the master archive `D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip` remain completely untouched.
- **Evidence-based reporting:** All claims, dimensions, and metrics are physically measured and verified against artifacts on disk.
- **No data fabrication:** The absence of verified Class 0 (normal skin) controls is explicitly documented as a hard blocker for binary model training.

---

## 2. Protected Asset Integrity Verification

Every protected asset and baseline artifact was programmatically verified via SHA-256 cryptographic hashing and byte-level inspection:

| Protected Asset | File Path | Size (Bytes) | Last Modified Timestamp | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Model** | [`models/best_model.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/best_model.keras) | 33,486,076 | 2026-10-15 21:50:51 | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **Intact / Verified** |
| **Augmented Model v1** | [`models/isic_augmented_v1.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_augmented_v1.keras) | 33,477,861 | 2026-10-26 12:18:27 | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **Intact / Verified** |
| **ISIC Finetuned Model** | [`models/isic_finetuned.keras`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/isic_finetuned.keras) | 33,477,534 | 2026-10-23 00:39:24 | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **Intact / Verified** |
| **OOD Gate Parameters** | [`models/ood_gate.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/ood_gate.npz) | 6,102,303 | 2026-10-26 10:30:41 | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **Intact / Verified** |
| **HAM Splits Manifest** | [`outputs/splits.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/splits.csv) | 1,259,807 | 2026-10-15 21:23:43 | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **Intact / Verified** |
| **Combined Splits Manifest** | [`outputs/combined_splits.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/combined_splits.csv) | 2,556,961 | 2026-10-23 00:11:18 | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **Intact / Verified** |
| **Active Config** | [`config.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/config.json) | 1,985 | 2026-10-26 15:46:08 | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **Intact / Verified** |
| **Master ZIP Backup** | [`D:/DOWNLOADS D/SkinDiseaseAI_v1_20261027.zip`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI_v1_20261027.zip) | 1,506,237,677 | 2026-10-27 10:10:21 | `11badbdd6b32140e55660c7ea2d3a499a3ba197ea222d29cb06b5bdaabe9bacd` | **Locked Master Backup** |

---

## 3. Architecture & Model Specifications

All existing disease classification models are based on **EfficientNetB0**:

1. **Input Dimensions:** `(224, 224, 3)` with standard 3-channel RGB image representation.
2. **Backbone:** `tf.keras.applications.EfficientNetB0` pre-trained on ImageNet.
   - Built-in internal rescaling and normalization layer (`Normalization` / `Rescaling(scale=1/255.0)`).
   - Expected input scale: raw floats in `[0.0, 255.0]`.
3. **Classification Head:**
   - Global Average Pooling 2D (`GlobalAveragePooling2D`).
   - Dropout layer with rate `0.30`.
   - Final classification layer: Dense with 7 units and `softmax` activation.
4. **Target Classes & Index Mapping:**
   Verified from [`models/class_indices.json`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/class_indices.json):
   ```json
   {
     "akiec": 0,
     "bcc": 1,
     "bkl": 2,
     "df": 3,
     "mel": 4,
     "nv": 5,
     "vasc": 6
   }
   ```
   - `akiec`: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (malignant/pre-malignant)
   - `bcc`: Basal cell carcinoma (malignant)
   - `bkl`: Benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses) (benign)
   - `df`: Dermatofibroma (benign)
   - `mel`: Melanoma (malignant)
   - `nv`: Melanocytic nevi / typical moles (benign)
   - `vasc`: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage) (benign)

5. **Augmentation Layer Architecture (`isic_augmented_v1.keras`):**
   Implemented as an online Keras Sequential block chained before the backbone:
   - `RandomFlip(mode="horizontal")`: Conservative horizontal flip only (no vertical flip).
   - `RandomRotation(factor=(-0.0556, 0.0556))`: Rotation up to $\pm 20^\circ$.
   - `RandomZoom(height_factor=(-0.08, 0.08))`: Mild scale variation ($\pm 8\%$).
   - `RandomTranslation(factor=(-0.05, 0.05))`: Translation ($\pm 5\%$ horizontal and vertical).
   - Active strictly during training; acts as exact identity pass-through during validation and inference.

---

## 4. Pipeline & Module Audit

### A. Preprocessing Consistency Audit (`src/preprocessing.py`)
- **Training Preprocessing:** `_load_and_resize(path, label, img_size)` decodes raw image bytes via `tf.io.decode_image(channels=3)`, resizes using bilinear interpolation to `(224, 224)`, and casts to `tf.float32` keeping raw pixel values in `[0, 255]`.
- **Inference Preprocessing:** `pil_to_array(img, img_size)` opens via PIL, converts to RGB, resizes with bilinear filter to `(224, 224)`, and casts to `np.float32` in `[0, 255]`.
- **Finding:** Preprocessing is **100% mathematically consistent** between training and inference. Neither pipeline introduces inadvertent double-normalization (e.g. dividing by 255 twice).

### B. Dataset Loading & Split Integrity Audit (`src/dataset.py`, `src/isic.py`)
- **Lesion Grouping:** In HAM10000, `split_by_lesion` partitions by `lesion_id` using `GroupShuffleSplit`.
- **Split Breakdown (HAM10000):**
  - Training: 7,011 images
  - Validation: 1,509 images
  - Test: 1,494 images (Frozen test partition)
- **ISIC 2019 Integration:**
  - In `combined_splits.csv`, 11,546 compatible ISIC images were added to training and validation splits.
  - Zero ISIC images were added to the test split. The test split is 100% preserved from the original HAM10000 benchmark.
  - Duplicate exclusion: 11,420 ISIC images that duplicated or near-duplicated HAM10000 images were explicitly excluded.

### C. Input Gate Audit (`src/gate.py`, `models/ood_gate.npz`)
- **Mechanism:** Mahalanobis distance calculated against an empirical Gaussian distribution fitted on 1,280-dimensional ImageNet embeddings of 4,000 dermoscopy training images.
- **Parameters on Disk:**
  - Centroid vector $\mu \in \mathbb{R}^{1280}$.
  - Precision matrix $\Sigma^{-1} \in \mathbb{R}^{1280 \times 1280}$.
  - Distance threshold: `thr = 2673.0803` (selected on validation dermoscopy images to accept 97.0%).
- **Quality limits:** Luminance $[50.43, 245]$, Blur variance $\ge 4.32$, dark fraction $\le 0.67$, saturation fraction $\le 0.07$.
- **Finding:** The gate is tightly calibrated for **dermoscopy close-ups**. When presented with non-contact smartphone photography, the domain shift pushes the Mahalanobis distance to $d \approx 5755 \gg 2673$, causing false rejection of valid smartphone skin photos.

---

## 5. Summary of Baseline Test Performance

Evaluated on the preserved, frozen HAM10000 test set (1,494 images; 187 melanomas):

| Model Artifact | Overall Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Mel Recall | Mel Precision | False Alarms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`best_model.keras` (Original)** | 67.60% | 63.49% | 0.5327 | 0.7048 | 53.48% (100/187) | 35.59% | 181 / 1,307 |
| **`isic_finetuned.keras`** | 70.48% | 66.94% | 0.5591 | 0.7284 | 41.71% (78/187) | 50.65% | 76 / 1,307 |
| **`isic_augmented_v1.keras`** | 72.29% | 65.12% | 0.5785 | 0.7428 | 37.97% (71/187) | 56.35% | 55 / 1,307 |
| **Ensemble (Orig + Aug v1, Argmax)** | 72.76% | 67.35% | 0.5907 | 0.7461 | 49.20% (92/187) | 48.68% | 97 / 1,307 |
| **Ensemble (Calibrated $t=0.347$)** | 71.08% | 69.21% | 0.5840 | 0.7380 | 54.01% (101/187) | 41.74% | 141 / 1,307 |

---

## 6. Critical Findings & Discrepancies

1. **The Melanoma Sensitivity vs. Accuracy Paradox:**
   As training progressed from `best_model.keras` to `isic_augmented_v1.keras`, headline accuracy rose by $+4.69\%$ ($67.60\% \to 72.29\%$) and Macro F1 rose from $0.533 \to 0.578$. However, **melanoma recall collapsed from $53.48\% \to 37.97\%$** (missing 116 out of 187 melanomas at default argmax). The model became conservative to minimize false positives on the dominant `nv` class (46 vs 150 false alarms), pushing true melanomas into `bkl` (60 missed melanomas) and `nv` (34 missed melanomas).
2. **Missing Class 0 Controls (Primary Blocker):**
   Neither HAM10000, ISIC 2019, Stanford DDI, nor Google SCIN provides verified normal-looking skin controls. Training a Normal vs. Disease binary classifier remains strictly **blocked** until the Option A volunteer collection (50 consenting subjects) is conducted.
3. **Severe Support Disparity on Minority Classes:**
   In the frozen test set, `df` has only **7 images** and `vasc` has only **21 images**. Point estimates for `df` ($F_1 = 0.320$) and `vasc` ($F_1 = 0.826$) exhibit wide confidence intervals and must be interpreted with documented caution.
4. **Smartphone Domain Mismatch in Gate:**
   The OOD gate rejects smartphone photos due to training on dermoscopic contact-plate images. Recalibration is mandatory before production deployment of smartphone features.
