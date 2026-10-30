# Dataset Combination & Training Status Verification Report

**Date:** 2026-10-30  
**Investigation Mode:** Read-Only Technical Audit  
**Target Directory:** `C:\Users\USER\Downloads\normal skin dataset`  
**Workspace:** `d:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI`  
**Safety Classification:** Educational research prototype — NOT a medical device  

---

## 1. Executive Conclusion

### **VERDICT: NOT COMBINED (TRAINED SEPARATELY OR NOT TRAINED)**

Based on an exhaustive programmatic, architectural, and artifact audit across the dataset directories, dataset manifests, training scripts, model weights, and experimental logs:

1. **NO DATASETS WERE COMBINED:**  
   The datasets inside `C:\Users\USER\Downloads\normal skin dataset` were **NEVER combined** into a single dataset or model.
2. **HAND DATASET (`archive (3)\Hands\Hands`): TRAINED SEPARATELY:**  
   The 11,076-image 11k Hands dataset was used **exclusively and in isolation** to train the one-class normal-hand appearance anomaly detector (`hand_anomaly_baseline.npz` and Stage G Candidate `candidate_1b_l2_mahalanobis_k32.npz`).
3. **FACIAL COSMETIC SKIN TYPES (`archive (2)\Oily-Dry-Skin-Types`): NOT TRAINED:**  
   The 3,152 facial close-up images with cosmetic sebum labels (`dry`, `normal`, `oily`) were audited in Stage A and **explicitly excluded**. Not a single image was ever loaded into any manifest, feature extractor, training script, or model.
4. **FACIAL CONDITION DATASET (`archive (3)\1`–`15`): NOT TRAINED:**  
   The 45 facial portrait images across folders `1` through `15` were audited in Stage A, identified as non-hand facial portraits, and **never trained**.
5. **ECZEMA & TINEA COLLECTION (`archive (4)`): NOT TRAINED:**  
   The 2,147 clinical lesion images across body sites (arms, legs, feet, torso) were audited in Stage A, identified as a fatal confounding shortcut risk, and **never trained**.
6. **DERMOSCOPIC 7-CLASS DISEASE MODELS (`best_model.keras`, etc.): COMPLETELY INDEPENDENT:**  
   The existing production lesion models were trained strictly on dermoscopic close-ups from **HAM10000** and **ISIC 2024**, completely disjoint from any folder in `C:\Users\USER\Downloads\normal skin dataset`.

---

## 2. Comprehensive Inventory of `C:\Users\USER\Downloads\normal skin dataset`

An exhaustive recursive inspection was conducted across all files and subdirectories:

| Archive / Subdirectory | Dataset Identity / Provenance | Total Images | Modality & Anatomical Site | Class Labels | Image Resolution & Format | Training Status |
|:---|:---|:---:|:---|:---|:---:|:---:|
| **`archive (2)\Oily-Dry-Skin-Types`** | Roboflow Cosmetic Skin Types | **3,152** | Facial close-ups (forehead, cheek, nose, chin) | `dry` (758)<br>`normal` (1,274)<br>`oily` (1,120) | $640 \times 640$, RGB JPEG | **NOT TRAINED**<br>(Excluded in Stage A) |
| **`archive (3)` (Folders `1`–`15`)** | Facial Skin Condition Dataset | **45** | Facial portraits (15 participants, 3 views each: front, right, left) | Individual Subject Folders `1` to `15` | Variable high-res ($960 \times 1280$ to $3088 \times 2316$), RGB JPEG | **NOT TRAINED**<br>(Excluded in Stage A) |
| **`archive (3)\Hands\Hands`** | 11k Hands (Afifi, 2019) | **11,076** | Dorsal and palmar human hands on white studio backdrop | Biometric metadata in `HandInfo.csv` (189 subjects) | $1600 \times 1200$, uniform RGB JPEG | **TRAINED SEPARATELY**<br>(Stage D Baseline & Stage G Candidate 1B) |
| **`archive (4)`** | Clinical Eczema vs Tinea Collection | **2,147** | Multi-site clinical lesions (torso, arms, legs, feet; ~8% hands) | `eczema` (1,122)<br>`tinea` (1,025) | Approx $720 \times 480$, RGB JPEG | **NOT TRAINED**<br>(Excluded in Stage A) |

### Detailed Breakdown by Modality:
- **Facial Images:** 3,197 images (`archive (2)`: 3,152; `archive (3)` folders 1–15: 45).
- **Hand Images:** 11,076 images (`archive (3)\Hands\Hands`).
- **Mixed Clinical Lesions:** 2,147 images (`archive (4)`: arms, legs, torso, feet, scalp).

---

## 3. Verification of Actual Training History

A comprehensive audit of repository manifests, training scripts, feature caches, and reports was performed to trace the origin of every model:

### 3.1 Dataset Manifests
1. **`data/experiments/normal_hand_v1/manifest_v1.csv` (11,076 rows):**
   - **Sources included:** 100% `archive (3)\Hands\Hands`.
   - **Sources excluded:** Exactly 0 rows from `archive (2)` (Oily-Dry), 0 rows from `archive (4)` (Eczema-Tinea), and 0 rows from `archive (3)` folders 1–15.
   - **Partitioning:** `train`: 7,655; `val`: 1,538; `test`: 1,639; `exploratory_evaluation` (quarantined): 243; `excluded` (duplicate): 1.
2. **`outputs/experiments/normal_hand_v1/stage_g/features_meta.csv` (2,758 rows):**
   - Features extracted and cached in `features_cache.npz` strictly originate from `archive (3)\Hands\Hands`. Zero facial images exist in the feature cache.
3. **`outputs/splits.csv` & `outputs/combined_splits.csv`:**
   - Contain strictly HAM10000 and ISIC dermoscopy records. Zero records from any archive in `normal skin dataset`.

### 3.2 Training & Feature Extraction Scripts
1. **`src/train.py` & `src/train_experiment.py`:**
   - Train the 7-class disease models on HAM10000 and ISIC dermoscopy. They have no loaders or references to hands, oily skin, or dry skin.
2. **`src/normal_hand/` Pipeline:**
   - Uses MediaPipe hand detection (`HandCropper`), MobileNetV3Small feature extractor, and `MahalanobisAnomalyDetector`.
   - Has zero code for cosmetic facial skin type classification or multi-class facial skin modeling.
3. **Audited Training Reports:**
   - In [`reports/experiments/normal_hand_dataset_audit.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/normal_hand_dataset_audit.md) (Stage A), the decision was documented:
     - `archive (2)\Oily-Dry-Skin-Types`: *"Unrelated to hand skin. Should be excluded from the hand model pipeline."*
     - `archive (4)`: *"Severe confounding risk... pairing archive (4) against archive (3) would result in a fatal confounding shortcut... excluded."*

---

## 4. Model Inventory & Training Provenance

| Model File & Exact Path | Architecture | Input Resolution & Preprocessing | Dataset Sources Used | Training Labels & Class Order | Sample Counts (Train / Val / Test) | Completion Status | Evaluation Metrics | Provenance Verification |
|:---|:---|:---|:---|:---|:---|:---:|:---|:---:|
| **`models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz`** | MobileNetV3Small + $L_2$ Normalization + PCA-Mahalanobis ($k=32$, $\lambda=0.05$) | $224 \times 224 \times 3$ (Hand Crop); normalized $[-1, +1]$; $L_2$ unit-sphere projection | 11k Hands (`archive (3)\Hands\Hands`) exclusively | One-class appearance density modeling (Single threshold: $t=6.931$) | Train: 1,919 (128 participants)<br>Val: 419 (28 participants)<br>Test: 420 (28 participants) | Fully fitted & calibrated | Val FPR: 5.01%<br>Test FPR: 7.86%<br>Accessory disparity: $0.98\times$ (near-perfect) | **CONFIRMED SEPARATE** (Fitted only on 11k Hands) |
| **`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`** | MobileNetV3Small + PCA-Mahalanobis ($k=48$, $\lambda=0.05$) | $224 \times 224 \times 3$ (Hand Crop); normalized $[-1, +1]$; raw 576-dim features | 11k Hands (`archive (3)\Hands\Hands`) exclusively | One-class appearance density modeling (Single threshold: $t=8.768$) | Train: 1,934 (129 participants)<br>Val: 419 (28 participants)<br>Test: 420 (28 participants) | Fully fitted & calibrated | Val FPR: 4.77%<br>Test FPR: 10.24%<br>Accessory disparity: $4.17\times$ | **CONFIRMED SEPARATE** (Fitted only on 11k Hands) |
| **`models/best_model.keras`** | DenseNet121 + GlobalAvgPool + Dense(7, softmax) | $224 \times 224 \times 3$; ImageNet standard normalization | HAM10000 dermoscopy close-ups | 7 Classes:<br>0: `akiec`<br>1: `bcc`<br>2: `bkl`<br>3: `df`<br>4: `mel`<br>5: `nv`<br>6: `vasc` | Train: 7,010<br>Val: 1,510<br>Test: 1,494 | Fully trained | HAM Val Macro F1: ~0.76<br>Balanced Acc: ~0.78 | **CONFIRMED SEPARATE** (Dermoscopy only; 0% hand/facial) |
| **`models/isic_finetuned.keras`** | DenseNet121 + GlobalAvgPool + Dense(7, softmax) | $224 \times 224 \times 3$; ImageNet standard normalization | HAM10000 + ISIC 2024 combined | 7 Classes (same as above) | Train: 16,546<br>Val: 3,465<br>Test: 1,494 | Fully trained | Balanced Acc: ~0.79<br>Melanoma Sensitivity: 84.1% | **CONFIRMED SEPARATE** (Dermoscopy only; 0% hand/facial) |
| **`models/ood_gate.npz`** | MobileNetV2 + Mahalanobis Gate | $224 \times 224 \times 3$; GlobalAvgPool (1280-dim) | HAM10000 dermoscopy training split | In-distribution dermoscopy vs out-of-distribution gate | Fitted on HAM10000 train split | Fully fitted | Rejects non-dermoscopy with 0.0% false acceptance on smoke fixtures | **CONFIRMED SEPARATE** (Dermoscopy gate only) |

---

## 5. Specific Inquiries Answered

### 1. Were normal hand images and normal facial images combined?
**NO.** Hand images and facial images were never combined in any manifest, feature extractor, or model. The hand model operates exclusively on dorsal and palmar crops from 11k Hands.

### 2. Were oily and dry facial images included in training?
**NO.** The oily and dry facial images (`archive (2)\Oily-Dry-Skin-Types`) were audited in Stage A and explicitly rejected because cosmetic sebum levels bear no relationship to hand anatomy or clinical skin disease. No model in the repository was trained on them.

### 3. Were all datasets used in the same model?
**NO.** There is no shared model. The dermoscopy disease classifier (`best_model.keras`) and the normal-hand anomaly detector (`candidate_1b...npz`) are completely separate pipelines with disjoint code, architectures, and datasets.

### 4. Does the existing trained model support hand images, facial images, or both?
- The normal-hand anomaly detector supports **strictly macro-camera hand images**. MediaPipe hand landmark detection at Step 1 rejects facial images as `NO_HAND_DETECTED` (`UNSUITABLE_OR_UNCERTAIN_INPUT`).
- The 7-class disease classifier supports **strictly dermoscopic close-up lesion images**.
- Neither model supports or classifies facial cosmetic skin types.

### 5. Are the models separate or shared?
**COMPLETELY SEPARATE.** The models have independent weights, separate file formats (`.keras` vs `.npz`), distinct input domains, and are invoked through separate branches in `src/predict.py`.

---

## 6. Audit Provenance & Verification Sign-Off

- **Investigation Character:** Read-only inspection.
- **Repository Integrity:** No code, models, configurations, manifests, or feature flags were modified during this investigation.
- **Quarantine Adherence:** Quarantined subjects (`27`, `570`, `1566`, and `1000000`) were never accessed or evaluated.
- **Report Generated:** [`reports/experiments/normal_hand_v1/dataset_combination_training_verification.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/normal_hand_v1/dataset_combination_training_verification.md).
