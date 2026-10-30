# Stage C: Learning-Task & Label-Validity Assessment Report

**Date:** October 29, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI — Normal-Looking Hand Skin Model  
**Artifact Evaluated:** [data/experiments/normal_hand_v1/manifest_v1.csv](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv)  
**Reference Audits:**  
* [reports/experiments/normal_hand_dataset_audit.md](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/normal_hand_dataset_audit.md)  
* [reports/experiments/normal_hand_v1/stage_b_manifest_report.md](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/experiments/normal_hand_v1/stage_b_manifest_report.md)  
**Current Milestone:** Stage C — Task Formulation & Label Validity Checkpoint (Pre-Implementation Gate)  

---

## 1. Executive Summary & Recommended Learning Paradigm

Following the dataset audit (Stage A) and subject-disjoint partitioning (Stage B), this assessment evaluates the mathematical, clinical, and machine-learning validity of candidate learning tasks using the 11k Hands dataset.

### Core Recommendation: Two-Stage Hybrid Architecture
The dataset **does not** contain verified pathological labels, but it **does** provide 10,832 clean, high-resolution photographs of visually typical hand skin across 185 unique individuals. The only technically sound, medically responsible formulation is a **Two-Stage Input-Suitability and Normal-Hand Density/Anomaly Pipeline**:

```
                       [ Uploaded Image ]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ Stage 1: Hand & Skin Suitability Gate        │
        │ • MediaPipe Landmark Extraction              │
        │ • Hand Presence & Landmark Bounding Box      │
        │ • Skin Tissue Chromaticity Filter            │
        │ (Rejects obvious non-hands, non-skin, blur)  │
        └──────────────────────┬───────────────────────┘
                               │ Hand Detected & Cropped
                               ▼
        ┌──────────────────────────────────────────────┐
        │ Stage 2: Normal Hand Skin Density / Anomaly  │
        │ • Deep Feature Embedding (MobileNetV3 / B0)  │
        │ • Mahalanobis / Density Modeling on Normal   │
        │   Reference Distribution (Train: 129 people) │
        │ • Calibrated Threshold on Held-Out Val Split │
        └──────────────────────┬───────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [ In-Distribution ]           [ Out-of-Distribution ]
    "Consistent with normal-        "Differs from reference
     looking hand skin baseline"     normal hand distribution"
     (Cautionary baseline check)     (Lesion, artifact, lighting)
```

### Explicit Negative Verdict on Supervised Binary "Normal vs. Disease"
**A supervised binary classifier (`normal_skin` vs. `disease`) CANNOT be trained from this dataset alone.** Attempting to train a supervised binary classifier using the four irregular participants or by pairing with external non-hand datasets would be scientifically invalid and create catastrophic confounding shortcuts.

---

## 2. Label Validity Analysis: Why `irregularities` is NOT a Medical Disease Label

The 11k Hands dataset was compiled by Mahmoud Afifi (2019) at York University for **biometric identity verification, gender recognition, and biometric authentication**, not for dermatological research.

### Critical Deficiencies of the `irregularities` Field
1. **Absence of Clinical Examination:** Images were collected in a computer vision laboratory by non-medical personnel. No board-certified dermatologist examined the participants, no biopsies were taken, and no clinical diagnostic codes (ICD-10/ICD-11) exist.
2. **Extreme Participant Clumping ($N=4$):** Although 164 images carry `irregularities = 1`, all 164 images originate from **only 4 individuals**:
   * **Subject `27`** (male, 23, fair): 21 images exhibiting a physical scar / surface irregularity.
   * **Subject `570`** (female, 21, medium): 21 images with a focal hyperpigmented patch / birthmark.
   * **Subject `1566`** (male, 21, medium): 99 images showing epidermal scratches / healed marks.
   * **Subject `1000000`** (male, 70, dark): 23 images showing aged skin texture, prominent veins, and senile purpura/lentigines.
3. **Severe Confounder Consequence:** If a classifier were trained using `irregularities \in {0, 1}`, the model would not learn a generalized pattern of "skin pathology"; it would simply memorize the biometric features, skin tones, and lighting signatures of these 4 specific individuals.
4. **Conclusion:** In `manifest_v1.csv`, `irregularities` must remain strictly categorized as a **visual surface anomaly flag**, not a medical label.

---

## 3. Systematic Task Suitability Assessment

| Candidate Learning Task | Feasibility with Manifest v1 | Clinical Rigor & Transparency | Recommendation | Justification & Risk Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Option A: Normal-Hand Appearance Modeling** | **High** | **Strong** | **RECOMMENDED (Stage 2)** | Feasible because the `train` partition provides 7,655 clean images across 129 people. Learns the visual manifold of typical hand skin without asserting medical health. |
| **Option B: Supervised Normal vs. Disease Classifier** | **Zero (alone)** | **Scientifically Invalid** | **REJECTED on this dataset** | No verified lesion data exists. Pairing with `archive (4)` creates a fatal body-site shortcut (hands on paper vs. torso on hospital bed). |
| **Option C: Normal-Hand Anomaly / Density Detection** | **High** | **Strong** | **RECOMMENDED (Companion to A)** | Learns a compact density/Mahalanobis boundary around normal hand embeddings. Anomaly scores flag unfamiliarity, avoiding unsupported diagnostic claims. |
| **Option D: Hand & Skin Suitability Gate** | **Very High** | **High** | **RECOMMENDED (Stage 1)** | Structural input check (MediaPipe landmark validation and skin chromaticity) to verify that an uploaded image contains a usable hand before invoking neural feature extractors. |

---

## 4. Confounder and Shortcut Analysis

Any model trained on this collection will inherit the biases of its physical capture protocol unless actively mitigated:

### 1. The White Background Shortcut
* **Physical Setup:** 100% of images were captured flat on an identical white matte background board ($R=253.6, G=253.4, B=253.2$).
* **Mechanism of Failure:** A convolutional neural network trained on full 1600×1200 frames will quickly latch onto the white background border frequency rather than skin features.
* **Mitigation:** Uncropped full-frame images must **never** be fed to the model. Preprocessing must crop directly to the hand bounding box or extract localized skin tissue patches.

### 2. Accessories & Jewellery Confounders
* **Prevalence:** **`3,211` images (28.99%)** contain rings, watches, or bracelets.
* **Mechanism of Failure:** Metallic specular reflections, watchbands, and gemstones could be mistaken for lesions or trigger false out-of-distribution alerts.
* **Mitigation:** The model must evaluate palm and dorsal skin regions with crop margins that minimize wristwatch dominance.

### 3. Fingernail Polish Confounders
* **Prevalence:** **`1,067` images (9.63%)** contain visible nail polish (bright red, dark enamel, patterns).
* **Mechanism of Failure:** Pigmented polish significantly shifts local chrominance histograms, which could cause a skin filter or texture estimator to misclassify distal phalanges.

### 4. Demographic & Age Bias
* **Age Profile:** Mean age is **22.1 years** (median 21.0, range 18–75).
* **Demographic Limitation:** Geriatric hands—which naturally feature thinning epidermis, prominent dorsal veins, solar elastosis, and senile lentigines—are virtually absent from the baseline cohort (Subject 1000000 was the only 70-year-old, and was quarantined).
* **Real-World Impact:** The model will exhibit higher false-anomaly rates on elderly hands. This limitation must be explicitly documented in user-facing warnings.

---

## 5. Proposed Preprocessing, Cropping, and Shortcut Verification Plan

To ensure the model learns skin texture rather than background artifacts, the following preprocessing pipeline is specified for Stage D:

### Preprocessing Pipeline: Hand Landmark Bounding-Box Extraction
1. **Landmark Detection:** Execute MediaPipe Hands on the 1600×1200 image to extract the 21 3D hand landmarks.
2. **Bounding Box Computation:** Calculate $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ from landmarks, adding a calibrated $10\%$ padding margin.
3. **Region Normalization:** Crop the hand region and resize to standard neural network input dimensions ($224 \times 224 \times 3$).
4. **Fallback for Edge Cases:** If landmark confidence is borderline, retain the largest detected skin contour.

```
Original Image (1600x1200)               Preprocessed Model Input (224x224)
┌─────────────────────────────────┐      ┌─────────────────────────┐
│   White Background Board        │      │ Hand Region Only        │
│          ┌──────────┐           │      │ (White borders excluded,│
│          │  Hand    │           │ ───► │  focus on dorsal/palmar │
│          │  Region  │           │      │  skin texture)          │
│          └──────────┘           │      └─────────────────────────┘
└─────────────────────────────────┘
```

### Programmatic Shortcut Verification Checks
Before accepting any Stage D model, the following empirical tests must be conducted:
1. **Shuffled / Synthetic Background Invariance:** Replace the surrounding background with synthetic Gaussian noise or neutral gray; verify that the embedding vector remains stable ($\Delta < 2\%$).
2. **Saliency / Grad-CAM Attention Map:** Inspect feature attribution maps to confirm that gradients are concentrated on the dorsal/palmar skin tissue, not on border corners or background pixels.

---

## 6. Formal Evaluation Protocol

The evaluation plan preserves complete subject separation and strictly protects the exploratory evaluation partition:

### A. Partition Allocation
* **`train` (129 participants, 7,655 images):** Used strictly for fitting the feature extractor and estimating baseline distribution statistics (mean vector $\boldsymbol{\mu}$, covariance matrix $\boldsymbol{\Sigma}$).
* **`val` (28 participants, 1,538 images):** Used exclusively for hyperparameter selection, dimensionality reduction tuning (PCA/embedding dimensions), and **calibrating the anomaly decision threshold $t$** (e.g., 95th or 99th percentile of normal distances).
* **`test` (28 participants, 1,639 images):** Held out completely until final verification. Evaluates in-distribution false rejection rate ($1 - \text{Specificity}$) on unseen individuals.
* **`exploratory_evaluation` (4 participants, 243 images):** **Quarantined completely during model development.** Evaluated post-hoc as a stress test to determine: *Do individuals with notable visual irregularities naturally produce elevated anomaly distances compared to the held-out test cohort?*

### B. Reported Metrics
1. **Mahalanobis Distance / Anomaly Score Distribution:** Percentiles (Min, 25%, Median, 75%, 90%, 95%, 99%, Max) across `train`, `val`, `test`, and `exploratory_evaluation`.
2. **In-Distribution False Rejection Rate:** Proportion of normal test images flagged as anomalous at the calibrated validation threshold.
3. **Exploratory Detection Rate:** Proportion of quarantined irregular images flagged as differing from the baseline manifold.

---

## 7. Requirements for a Future Medically Meaningful "Normal vs. Disease" Study

To transition from an experimental normal-appearance baseline to a clinically valid hand disease classifier, the following external data assets would be required:

1. **Certified Hand Lesion Cohort:** A curated dataset of common hand dermatoses (e.g., hand eczema/pompholyx, tinea manuum, psoriasis vulgaris, palmar pustulosis, scabies, actinic keratoses, and melanoma of the nail unit/acral surfaces).
2. **Dermatological Ground Truth:** Every lesion must be verified by a board-certified dermatologist via dermoscopy, clinical photographs, or histopathology.
3. **Matched Capture Modality:** Lesion images must be clinical macro-photographs of human hands taken under diverse real-world lighting, not dermatoscope close-ups or non-hand body locations.
4. **Demographic Parity:** Cohorts must span Fitzpatrick skin types I through VI and age brackets from 18 to 85+ years to prevent skin-tone or age confounding.

---

## 8. Prerequisites and Gates Before Stage D

The following gates must be cleared before starting Stage D baseline model development:

* [x] **Gate 1:** Stage A dataset audit completed and documented.
* [x] **Gate 2:** Stage B manifest generated with strict subject-level disjointness and SHA-256 sealed.
* [x] **Gate 3:** Stage C task assessment completed, disclaiming unsupported medical disease detection and recommending the two-stage suitability + anomaly pipeline.
* [ ] **Gate 4:** Explicit user approval of Stage C findings and authorization to begin Stage D.
* [ ] **Gate 5:** Implementation of isolated hand bounding-box cropping utilities in `src/normal_hand/`.
* [ ] **Gate 6:** Hardware and CPU runtime check verified (native Windows TensorFlow CPU execution; lightweight backbone MobileNetV3 selected).

---

**STOPPED at Stage C Checkpoint.** This concludes the learning-task and label-validity assessment. Awaiting your approval before proceeding to Stage D (Baseline Model Architecture and Implementation).
