# 📊 SkinDiseaseAI — Seven-Class Lesion Model Improvement Results

**Project:** SkinDiseaseAI — Complete Model Accuracy Improvement and Generalization Project  
**Author:** Senior Machine Learning Engineer & Medical CV Specialist  
**Date:** October 29, 2026  
**Document:** `reports/model_improvement/seven_class_improvement_results.md`  
**Evaluation Cohort:** Preserved, Frozen HAM10000 Test Split (1,494 images; 187 melanomas)  
**Execution Context:** Evaluated strictly on the frozen test partition without hyperparameter retuning  

---

## 1. Executive Summary & Progression

This report details the comparative performance of the three trained seven-class EfficientNetB0 models and the calibrated ensemble configuration on the preserved, frozen HAM10000 test set:

1. **`best_model.keras` (Original Baseline):** Trained exclusively on HAM10000 training partition ($N = 7,011$) without data augmentation or class weighting.
2. **`isic_finetuned.keras`:** Fine-tuned on combined HAM10000 + ISIC 2019 data ($N = 16,546$) without data augmentation.
3. **`isic_augmented_v1.keras`:** Fine-tuned on combined data with conservative online data augmentation (flips, rotations, zooms) and balanced capped class weighting (cap = 5.0).
4. **`Ensemble (Orig + Aug v1)`:** Probability-averaged ensemble combining baseline feature representations with augmented representations.
5. **`Calibrated Threshold Model`:** Post-hoc decision-threshold calibration tuned on validation data ($t^* = 0.347$ for ensemble, $t^* = 0.233$ for aug v1) to safeguard melanoma recall.

---

## 2. Comparative Performance Matrix on Frozen Test Split

The table below summarizes measured, reproducible metrics on the 1,494 test images:

| Model Architecture & Configuration | Overall Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Mel Recall (Sensitivity) | Mel Precision | False Alarms (Non-mel flagged) | Macro ROC-AUC | Macro PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`best_model.keras` (Original Baseline)** | 67.60% | 63.49% | 0.5327 | 0.7048 | 53.48% (100/187) | 35.59% | 181 / 1,307 (13.8%) | 0.9255 | 0.6187 |
| **`isic_finetuned.keras`** | 70.48% | 66.94% | 0.5591 | 0.7284 | 41.71% (78/187) | 50.65% | 76 / 1,307 (5.8%) | 0.9288 | 0.6412 |
| **`isic_augmented_v1.keras` (Argmax)** | 72.29% | 65.12% | 0.5785 | 0.7428 | 37.97% (71/187) | **56.35%** | **55 / 1,307 (4.2%)** | **0.9334** | **0.6645** |
| **Ensemble (Orig + Aug v1, Argmax)** | **72.76%** | 67.35% | **0.5907** | **0.7461** | 49.20% (92/187) | 48.68% | 97 / 1,307 (7.4%) | 0.9325 | 0.6610 |
| **⭐ Ensemble (Calibrated $t=0.347$)** | 71.08% | **69.21%** | 0.5840 | 0.7380 | **54.01% (101/187)** | 41.74% | 141 / 1,307 (10.8%) | 0.9325 | 0.6610 |
| **Aug v1 (Calibrated $t=0.233$)** | 69.61% | 67.89% | 0.5798 | 0.7291 | **54.01% (101/187)** | 37.00% | 172 / 1,307 (13.2%) | **0.9334** | **0.6645** |

---

## 3. Per-Class F1-Score Breakdown

| Class | Support ($N_{\text{test}}$) | Reliability Status | `best_model.keras` F1 | `isic_finetuned.keras` F1 | `isic_augmented_v1.keras` F1 | Delta (Aug v1 vs Baseline) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`akiec`** | 63 | Adequate ($\ge 20$) | 0.4706 | 0.4364 | 0.4651 | -0.0055 |
| **`bcc`** | 68 | Adequate ($\ge 20$) | 0.5972 | 0.6383 | **0.6471** | **+0.0498** |
| **`bkl`** | 152 | Adequate ($\ge 20$) | 0.4977 | 0.4736 | 0.4741 | -0.0236 |
| **`df`** | 7 | **Low support ($<20$)** | 0.2051 | 0.2941 | **0.3200** | **+0.1149** (Wide CI) |
| **`mel`** | 187 | Adequate ($\ge 20$) | 0.4274 | 0.4575 | **0.4537** | **+0.0263** |
| **`nv`** | 996 | Dominant (66.7%) | 0.8139 | 0.8449 | **0.8634** | **+0.0494** |
| **`vasc`** | 21 | Borderline ($\ge 20$) | 0.7170 | 0.7692 | **0.8261** | **+0.1091** |
| **Macro Average** | **1,494** | — | **0.5327** | **0.5591** | **0.5785** | **+0.0458** |
| **Weighted Average** | **1,494** | — | **0.7048** | **0.7284** | **0.7428** | **+0.0380** |

---

## 4. Deep Diagnostic Error Analysis: The Melanoma Trade-Off

### A. The Melanoma Sensitivity Drop
While overall accuracy improved significantly ($67.60\% \to 72.29\%$), default argmax prediction in `isic_augmented_v1.keras` exhibited a severe clinical vulnerability: **melanoma recall dropped from $53.48\%$ down to $37.97\%$** (missing 116 out of 187 test melanomas).

### B. Confusion Patterns for Missed Melanomas
Analyzing the confusion matrix reveals where the 116 missed melanomas were categorized:
- **60 melanomas misclassified as `bkl`** (Benign keratosis-like lesions).
  - *Clinical Mechanism:* Amelanotic or hypopigmented melanomas and lentigo maligna share irregular border patterns, reticular netting, and brownish keratotic surface features that neural networks readily confuse with benign seborrheic keratoses.
- **34 melanomas misclassified as `nv`** (Melanocytic nevi).
  - *Clinical Mechanism:* Early-stage malignant melanomas frequently arise within or adjacent to pre-existing dysplastic nevi, presenting similar pigment architectures.
- **12 melanomas misclassified as `akiec`** (Actinic keratoses).
- **7 melanomas misclassified as `bcc`** (Basal cell carcinoma).
- **3 melanomas misclassified as `df`** (Dermatofibroma).

```
Destinations of 187 True Melanomas in isic_augmented_v1 (Argmax):
Correct (mel)    [███████████████████                               ] 71 (38.0%)
Misclassified bkl[████████████████                                  ] 60 (32.1%)
Misclassified nv [█████████                                         ] 34 (18.2%)
Misclassified oth[█████                                             ] 22 (11.8%)
```

### C. Resolution via Calibrated Follow-Up Flagging
In medical triage, a single argmax decision boundary is insufficient. By tuning a dedicated screening threshold $t^*$ strictly on validation data:
- Setting $t^* = 0.347$ on the ensemble restored test melanoma recall to **54.01% (95% CI: [47.2%, 60.8%])** while maintaining melanoma precision at **41.74%** and limiting false alarms to **10.8%** of non-melanoma lesions.
- This represents an optimal Pareto trade-off between sensitivity and alert fatigue, fully integrated into the Streamlit application.

---

## 5. Computational & Hardware Efficiency

All evaluations and inference pipelines were benchmarked on the host hardware:
- **Parameter Count:** 4,058,975 parameters (~4.06M weights).
- **Disk Footprint:** 33.48 MB (`.keras` format).
- **Batch Inference Throughput:** ~38 images per second on multi-core CPU (AVX2/oneDNN).
- **Single-Image Latency:** ~28–36 ms per forward pass.
- **Peak RAM Utilization:** ~2.1 GB during full test split evaluation.
