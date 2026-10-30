# Stage R — Clinical Skin Appearance Model: Controlled Training & Bias Report

**Date:** 2026-10-30  
**Author:** Senior Computer Vision & Deep Learning Engineer  
**Project:** SkinDiseaseAI — Clinical Skin Appearance Pipeline  
**Model Artifact:** `models/clinical_skin_stage_r/clinical_skin_stage_r_best.keras`  
**Output Artifacts:** `outputs/clinical_skin_stage_r/`  
**Safety Classification:** Educational research prototype — NOT a medical diagnostic device  

---

## 1. Executive Summary

In Stage R, we conducted a controlled experimental training run of the Clinical Skin Appearance Model using the Stage Q dataset manifest (`manifest_all_local_clinical.csv`).

### Key Findings:
1. **Model Convergence:** MobileNetV3Small trained with binary cross-entropy and class weighting reached validation ROC-AUC = 1.000, PR-AUC = 1.000, Sensitivity = 86.96%, and Specificity = 100.0% across 2,358 validation images (2,036 normal hands, 322 clinical conditions).
2. **Critical Domain Bias Confirmed (Background Confounder):** A control logistic regression model trained **strictly on 10x10 corner background pixels** achieved an ROC-AUC of **1.000** and accuracy of **100.0%**. The 11k Hands dataset was captured on pure white studio backgrounds (mean RGB [252.2, 252.8, 252.4]), whereas the Eczema/Tinea dataset was captured in natural ambient clinical settings (mean RGB [74.0, 56.0, 52.1]).
3. **Implication:** The high in-distribution validation performance is heavily driven by **class-source domain features and background illumination** rather than genuine skin micro-textures.
4. **Generalization Benchmark (Stanford DDI):** On the independent external Stanford DDI cohort (656 images of focal lesions across Fitzpatrick Skin Types I–VI), the model predicted an abnormal condition rate of only **35.67%**, confirming that the model cannot be deployed as a general clinical skin disease detector.
5. **Architectural Recommendation:** Hand and skin region cropping, strict multi-stage input validation, out-of-distribution rejection, and explicit uncertainty thresholds are mandatory before presenting any prediction to users.

---

## 2. Dataset Selection & Cohort Definition

Per strict experimental requirements:
- **Normal Class (Class 0):** 11k Normal Hands filtered strictly to `irregularities = 0` (clean, normal skin). The 164 quarantined irregularity images (bandaids, scars, rings, visible blemishes) were **excluded**.
- **Potentially Abnormal Class (Class 1):** Verified clinical Eczema vs Tinea images (2,147 images).
- **Excluded from Training:** Stanford DDI was strictly preserved as an independent external test set. Benign lesions were never relabeled as normal skin.
- **Partitioning:** Preserved patient-level splits from Stage Q (Train: 8,674, Val: 2,358, Test: 2,191) with zero patient leakage.

| Dataset Split | Normal Hands (`target=0`) | Clinical Abnormal (`target=1`) | Total Images |
|:---|:---:|:---:|:---:|
| **Train** | 7,172 | 1,502 | 8,674 |
| **Validation** | 2,036 | 322 | 2,358 |
| **Test** | 1,868 | 323 | 2,191 |
| **Independent External (DDI)** | 0 | 656 | 656 |

---

## 3. Domain Bias & Background Leakage Investigation

To determine whether the classifier was learning disease features or domain shortcuts, we extracted background color features from 10x10 corner pixels across normal and disease images.

### Quantitative Background Profile:
- **Normal Hands Background:** Mean RGB = `[252.2, 252.8, 252.4]`, Standard Deviation = `3.75` (Uniform studio white).
- **Disease Skin Background:** Mean RGB = `[74.0, 56.0, 52.1]`, Standard Deviation = `31.52` (Ambient room/clinical lighting).

### Background-Only Control Classifier:
- A linear logistic regression model trained solely on `[Mean_R, Mean_G, Mean_B, Corner_Variance]` achieved:
  - **Background-Only ROC-AUC:** `1.000`
  - **Background-Only Accuracy:** `100.0%`

> [!CAUTION]
> **Scientific Finding:** This confirms severe class-source confounding. Without hand cropping or background removal, standard neural networks will inevitably use background illumination as a trivial shortcut. We explicitly document this limitation and do not claim that the model has learned general skin pathology.

---

## 4. Controlled Model Architecture & Training Parameters

- **Backbone:** MobileNetV3Small (ImageNet pretrained weights, frozen feature extractor fine-tuned end-to-end).
- **Classification Head:** Dense(128, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dense(1, Sigmoid).
- **Input Resolution:** $224 \times 224 \times 3$ RGB (raw range $[0, 255]$ with internal MobileNetV3 scaling).
- **Loss Function:** Binary Cross-Entropy with balanced inverse-frequency class weights:
  - Class 0 weight: $0.605$
  - Class 1 weight: $2.887$
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-4}$), batch size 32, with early stopping on validation ROC-AUC.

### Training Convergence:
| Epoch | Train Loss | Train ROC-AUC | Train Acc | Val Loss | Val ROC-AUC | Val Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0542 | 0.9991 | 98.84% | 0.1526 | 1.0000 | 91.52% |
| 2 | 0.0011 | 1.0000 | 100.0% | 0.0386 | 1.0000 | 98.22% |
| 3 | 0.0004 | 1.0000 | 100.0% | 0.0079 | 1.0000 | 99.79% |
| 4 | 0.0005 | 1.0000 | 99.99% | 0.0004 | 1.0000 | 100.0% |

---

## 5. In-Distribution Validation Performance

Evaluated on the patient-isolated Stage Q validation set (2,358 images):

| Metric | Score |
|:---|:---:|
| **ROC-AUC** | **1.0000** |
| **PR-AUC** | **1.0000** |
| **Sensitivity (Recall)** | **86.96%** |
| **Specificity** | **100.00%** |
| **Precision** | **100.00%** |
| **F1-Score** | **0.9302** |
| **True Negatives (TN)** | 2,036 |
| **False Positives (FP)** | 0 |
| **False Negatives (FN)** | 42 |
| **True Positives (TP)** | 280 |

---

## 6. Independent Domain Generalization Evaluation (Stanford DDI)

We evaluated the model against the independent Stanford DDI cohort (656 clinical close-ups of confirmed lesions, zero overlap with training):

- **Predicted Abnormal Rate ($t=0.50$):** **35.67%** (234 / 656)
- **Mean Abnormal Probability:** **0.3690**
- **Benign Lesions Mean Probability:** **0.3555**
- **Malignant Lesions Mean Probability:** **0.4071**
- **Subgroup Breakdown by Fitzpatrick Skin Tone (FST):**
  - FST I–II: Mean prob = `0.3769`
  - FST III–IV: Mean prob = `0.3476`
  - FST V–VI: Mean prob = `0.3859`

### Clinical Interpretation:
The low abnormal predicted rate (35.67%) on DDI demonstrates that clinical lesion images from diverse clinic environments lack the specific background signatures of the Eczema/Tinea training set. This further corroborates that the model cannot be interpreted as a general dermatological classifier.

---

## 7. Uncertainty and Unsupported Gate

To protect users against spurious predictions:
- **Low-Confidence Uncertainty Interval:** Any output probability $P \in [0.35, 0.65]$ is designated as **Uncertain**.
- **Domain Gate / OOD Check:** Images that fail skin coverage ($<15\%$), fail anatomical checks, or exhibit high feature Mahalanobis distances are designated as **Unsupported**.
- **Educational Notice:** A prediction of "normal-looking hand appearance" strictly indicates statistical similarity to the normal reference cohort, not clinical health.
