# 📐 SkinDiseaseAI — Stage B: Label Mapping and Split Plan (Revised & Finalized)
**Project:** SkinDiseaseAI — Normal-Looking Skin vs. Possible Skin Disease Classifier  
**Role:** Senior Machine Learning Engineer (Medical Computer Vision & Dataset Quality)  
**Date:** October 29, 2026  
**Status:** Stage B Plan Revised with User Review Corrections (Planning Only — No Code/Data Modifications)

---

## 1. Executive Summary & Objective

The objective of **Stage B** is to formulate an academically rigorous, statistically defensible, and medically honest architecture for a separate **Normal-Looking Skin vs. Possible Skin Disease** binary classifier, designed specifically for consumer smartphone and clinical photography.

This plan integrates the empirical findings from:
- [`reports/normal_disease/dataset_audit_report.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/normal_disease/dataset_audit_report.md) (Stage A Audit)
- [`reports/normal_disease/normal_skin_data_plan.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/normal_disease/normal_skin_data_plan.md) (Normal Skin Sourcing Plan)

```
                               ┌─────────────────────────────────┐
                               │     INCOMING SMARTPHONE IMAGE   │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   STAGE 1: ABSTENTION GATE      │
                               │   (Independent Suitability Check│
                               │    - Non-skin / OOD detection   │
                               │    - Focus, Lighting, Contrast  │
                               │    - Multi-hand screening)      │
                               └────────┬───────────────┬────────┘
                                        │               │
                            [Passed Quality]       [Failed Quality]
                                        │               │
                                        ▼               ▼
                        ┌────────────────────────┐  ┌────────────────────────┐
                        │   STAGE 2: CLASSIFIER  │  │ UNCERTAIN / UNSUITABLE │
                        │  Normal vs. Disease    │  │ (Abstain: No Disease   │
                        └───────┬────────┬───────┘  │  Scores Displayed)     │
                                │        │          └────────────────────────┘
                    ┌───────────┘        └───────────┐
                    ▼                                ▼
        ┌───────────────────────┐        ┌───────────────────────┐
        │  CLASS 0: NORMAL SKIN │        │ CLASS 1: DISEASE SKIN │
        │  "Normal-looking skin │        │ "Possible skin lesion │
        │   patch (no active    │        │  or abnormality       │
        │   lesion observed)"   │        │  detected"            │
        └───────────────────────┘        └───────────────────────┘
```

---

## 2. Proposed Pilot Collection (Option A: Normal Skin)

### A. Volunteer Cohort Design
- **Target Cohort:** **50 consenting volunteer participants** (initial pilot phase).
- **Governance:** Subject to institutional review / educational research ethics guidelines, formal informed consent, and strict privacy safeguards.
- **Anatomical Sampling Protocol:**
  - 4 anatomical regions per participant:
    1. Volar (inner) forearm
    2. Dorsal (outer) forearm
    3. Dorsum (back) of hand or palm
    4. Lower leg (anterior shin or lateral calf)
  - 3 standardized lighting environments per region:
    1. Natural diffused daylight (near window, non-direct sunlight)
    2. Warm indoor incandescent / warm LED (~3000K)
    3. Cool white commercial / fluorescent LED (~5500K–6500K)
  - 2 framing distances:
    1. Macro close-up (10–15 cm)
    2. Medium clinical framing (25–35 cm)

### B. Usable Image Yield Estimates
- **Raw Acquisition Estimate:**
  - 50 participants $\times$ 4 sites $\times$ 3 lights $\times$ 2 distances = **1,200 raw photos**.
- **Usable Yield after Quality Control (QC):**
  - Anticipating a realistic 20–25% rejection rate during automated quality checks (motion blur, specular reflections, focal slip):
  - **Conservative Usable Yield:** **~900 to 950 verified, high-quality normal skin patches**.

### C. The "Independence Fallacy" in Medical Computer Vision
> [!IMPORTANT]
> **Why multiple images from one person do NOT count as independent subjects:**  
> A critical flaw in medical machine learning is conflating the number of image frames ($M = 900$) with the true statistical sample size ($N = 50$). In human biology:
> - Melanin density, baseline skin undertone, epidermal thickness, subcutaneous capillary patterns, pore geometry, and follicle density are **intrinsic individual biological traits**.
> - If an algorithm trains on 15 photos of Person A across different lighting and tests on 5 photos of Person A, the model can simply **memorize Person A's personal biological signature**.
> - This produces high training and test accuracy that completely collapses when tested on an unseen Person B.
> 
> **Statistical Rule:**  
> The true statistical sample size for generalization is the number of **independent subjects ($N = 50$)**, not the number of image frames. Cross-validation and test evaluation must be clustered strictly at the subject level.

---

## 3. Label Definitions & Mapping Specifications

### Outcome 1 — Class 0: Normal-Looking Skin (Negative Class)
- **Eligibility:** Skin patches verified to be intact, unbroken, and free from active inflammatory, neoplastic, or infectious processes, collected under the approved 6-pillar protocol.
- **Physiological Variations Permitted:** Natural skin pores, fine vellus hair, and subtle non-dysplastic ephelides (freckles) are accepted as standard physiological characteristics.
- **Strict Medical Boundary:**
  > [!WARNING]
  > The label is explicitly designated **"Normal-looking skin patch"**, NOT "Medically normal person". The system must clearly report: *"No apparent lesion or abnormality observed in this sampled skin frame. This result does NOT rule out systemic skin disease, microscopic pathology, or lesions in unphotographed regions."*

---

### Outcome 2 — Class 1: Possible Skin Disease (Positive Class)
Images showing verified pathological lesions, inflammatory dermatoses, infections, or neoplasms.

#### A. Mapping Stanford DDI (Diverse Dermatology Images) to Class 1
- **Dataset Composition:** All 656 DDI images represent active clinical skin lesions requiring dermatological evaluation (171 malignant lesions, 485 benign lesions).
- **Mapping Strategy:**
  - `malignant == True` (171 images: melanoma, BCC, SCC, cutaneous lymphoma) $\longrightarrow$ **Class 1**.
  - `malignant == False` (485 images: seborrheic keratosis, melanocytic nevi, verruca vulgaris, cysts, dermatofibroma, acrochordon) $\longrightarrow$ **Class 1**.
- **Rationale & Inherent Limitations:** In a Normal vs. Disease binary triage setting, a patient presenting with an active lesion (even a benign growth like a wart, epidermal cyst, or seborrheic keratosis) presents a cutaneous abnormality distinct from clear unbroken skin. However, mapping benign lesions to "Class 1: Possible Disease" is an operational triage simplification. It does **not** imply that a benign mole is a dangerous disease.
- **License Compliance:** Stanford AIMI Research-Only Data Use Agreement allows non-commercial academic research and evaluation; prohibits re-identification and clinical deployment on patients.

#### B. Mapping Google SCIN to Class 1
- **Source Nature:** Crowdsourced smartphone images evaluated retrospectively by up to 3 dermatologists without biopsy confirmation. Condition weights represent aggregated differential confidence scores, not histopathological ground truth.
- **Proposed Heuristic Filtering Rule:**
  - **Filtering Threshold:** Retain cases where the top differential diagnosis in `weighted_skin_condition_label` has an aggregated weight $\ge 0.40$ AND `dermatologist_gradable_for_skin_condition == True`.
  - **Terminology Standard:** Retained cases must be documented as **"Dermatologist differential consensus cases"**, NEVER as "confirmed diagnoses".
- **Pathologies Included under Consensus Rule:**
  - Eczema / atopic dermatitis
  - Psoriasis vulgaris
  - Acne vulgaris and rosacea
  - Contact dermatitis (allergic and irritant)
  - Superficial fungal infections (tinea corporis, tinea pedis)
  - Viral exanthems and warts
- **Exclusions from Class 1:**
  - Cases marked `dermatologist_gradable_for_skin_condition == False` (ungradable, poor lighting, conflicting conditions) $\longrightarrow$ Excluded from training; routed to Abstention benchmarking.
  - User self-reported `related_category == 'LOOKS_HEALTHY'` $\longrightarrow$ Excluded from Class 0 (unverified layperson report).

#### C. Fitzpatrick17k Status
- **Quarantine:** Excluded from initial training until images are physically downloaded and verified. Metadata remains available for auxiliary reference.

---

### Outcome 3 — Abstention: `UNCERTAIN_OR_UNSUITABLE`
The model must support an automated abstention pathway rather than forcing ambiguous images into a binary choice.

> [!CAUTION]
> **Validation of Input Suitability Thresholds:**  
> Existing threshold values (such as the Mahalanobis distance threshold of 2673.08 from `ood_gate.npz` or the Laplacian blur threshold of 40.0) were fitted on dermoscopic images (HAM10000/ISIC) or synthetic quality checks.  
> **These thresholds must NOT be assumed universally reliable for smartphone photography.**  
> - In Stage C, the input suitability pipeline must be audited independently on real smartphone photos.
> - Empirical false rejection rates (on valid smartphone skin) and false acceptance rates (on non-skin images) must be measured and reported before finalizing gate thresholds.

| Trigger Condition | Target Mechanism | System Action |
| :--- | :--- | :--- |
| **Non-Skin / Out-of-Domain Content** | Mahalanobis Distance / OOD Embedding (calibrated for smartphone domain) | Rejection: *"Input is not recognized as skin tissue."* |
| **Severe Motion Blur** | Laplacian Variance (empirically tuned for phone cameras) | Rejection: *"Image is too blurry for reliable evaluation."* |
| **Extreme Lighting** | Mean luminance $< 35$ or $> 235$; Saturation fraction $> 0.25$ | Rejection: *"Image is underexposed or contains severe glare."* |
| **Low Contrast** | Contrast standard deviation $< 12.0$ | Rejection: *"Image lacks sufficient visual contrast."* |
| **Multi-Hand Interference** | MediaPipe Hand Landmark Count $> 1$ | Rejection: *"Multiple hands detected in frame."* |
| **Model Classification Uncertainty** | Prediction confidence $0.40 < P(\text{disease}) < 0.60$ | Abstention: *"Borderline / Uncertain. Clinical inspection required."* |

---

## 4. Cross-Dataset Compatibility & Confounder Analysis

Combining clinical images (DDI, SCIN) with volunteer smartphone images (Option A) creates substantial risks of **shortcut learning**.

```
                           CONFOUNDING THREAT ANALYSIS
                           
 [Option A: Normal Skin]                              [DDI: Disease Skin]
 - Smartphone camera (Samsung/OnePlus)                - Hospital clinical DSLR
 - Ambient home / window lighting                     - Exam-room surgical lighting
 - Indian / South Asian skin tones (FST IV-VI)       - California clinical cohort
 - Macro skin texture framing                         - Standardized clinical framing
                        \                                    /
                         ▼                                  ▼
                   ┌──────────────────────────────────────────────┐
                   │    CRITICAL RISK: SHORTCUT LEARNING          │
                   │ The CNN learns:                              │
                   │ "Smartphone sensor + Home light = Normal"    │
                   │ "Hospital DSLR + Medical light = Disease"    │
                   │ (Completely ignores actual skin pathology!)  │
                   └──────────────────────────────────────────────┘
```

### Confounder Identification & Mitigation Table

| Potential Confounder | Description of Risk | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Camera Sensor & ISP** | Model classifies CMOS sensor noise or color temperature rather than lesion pathology. | **Source Blending:** Class 1 must include consumer smartphone images from Google SCIN, not solely DSLR DDI images.<br>**Data Augmentation:** Apply random color jitter, Gaussian noise, and ISO grain simulation during training. |
| **Framing & Background** | Hospital exam table paper or wall visible in DDI; home background visible in volunteer photos. | **ROI Skin Cropping:** Preprocessing must tightly crop skin tissue (using skin segmentation masks) to eliminate all non-skin background artifacts. |
| **Resolution Mismatch** | DDI has large images ($1000\times1400$ px); volunteer photos may vary. | Standardize all inputs to identical bilinear resized $224\times224\times3$ tensors. |
| **Demographic Confounding** | If Normal skin is predominantly FST V–VI and Disease skin is predominantly FST I–III, the model will learn that dark skin is normal and light skin is diseased. | **Skin-Tone Stratified Sampling:** Enforce balanced representation of FST I–II, III–IV, and V–VI across both Class 0 and Class 1. |
| **DDI Missing Patient IDs** | `ddi_metadata.csv` lacks patient identifiers. Splitting DDI across train and test could leak multiple lesions from the same patient into both splits. | **Architectural Isolation:** Reserve Stanford DDI **entirely as an independent, frozen External Evaluation Benchmark**, keeping it 100% out of training and validation! |

---

## 5. Subject-Independent Partitioning Plan (70 / 15 / 15)

To guarantee scientific validity, all training and validation splits are partitioned strictly by **independent subjects**.

```
                             FULL DATASET POOL
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │                                                       │
         ▼                                                       ▼
[PRIMARY CORPUS: Subject IDs Known]             [EXTERNAL BENCHMARK: DDI]
- Option A Volunteers (SUBJ_001 to SUBJ_050)    - 656 Clinical Lesions
- Google SCIN (Grouped by case_id)              - Missing Patient IDs
         │                                                       │
         ▼ (Stratified GroupShuffleSplit by Skin Tone)           │ (Preserved 100% Frozen)
   ┌─────┬─────────────────┬─────┐                               │
   │     │                 │     │                               │
   ▼     ▼                 ▼     ▼                               ▼
[TRAIN: 70%]        [VAL: 15%]  [INTERNAL TEST: 15%]     [EXTERNAL BENCHMARK: 100%]
- 35 Subjects       - 7-8 Subj  - 7-8 Subjects           - All 656 DDI images
- SCIN 70% Cases    - SCIN 15%  - SCIN 15% Cases         - Zero training leakage
- Feature Learning  - Checkpoint- Internal Benchmark     - Evaluates Lesion-Domain
  & Class Weights     Selection   (Normal vs Disease)      Behavior Only (Not full accuracy)
```

### A. Partition Allocation Strategy
1. **Train Split (70% of Independent Subjects):**
   - 35 volunteer subjects (Option A) + 70% of validated SCIN cases.
   - Purpose: Backbone feature extraction, representation learning, and class weight calculation.
2. **Validation Split (15% of Independent Subjects):**
   - 7 to 8 volunteer subjects (Option A) + 15% of validated SCIN cases.
   - Purpose: Model checkpoint selection, learning rate decay triggers, and candidate decision threshold tuning.
3. **Internal Frozen Test Split (15% of Independent Subjects):**
   - 7 to 8 volunteer subjects (Option A) + 15% of validated SCIN cases.
   - Purpose: Evaluated **ONCE** at project completion to measure in-domain smartphone test accuracy, precision, recall, specificity, and F1 with 95% confidence intervals.
4. **External Frozen Benchmark (100% of Stanford DDI — 656 images):**
   - **Zero DDI images in training or validation.**
   - **Strict Evaluation Scope:**
     > [!IMPORTANT]
     > Because DDI contains **only lesions and zero normal controls**, DDI cannot measure true negative rate, specificity, or overall binary accuracy.  
     > DDI performance will be reported strictly as a **Lesion Detection Sensitivity Analysis across Dark and Light Skin Tones**, measuring whether the model correctly flags diverse clinical lesions without having seen DDI cameras during training.

### B. Leakage Prevention Protocol
1. **Strict Cluster Boundary:** No images, body sites, or lighting variations from `SUBJ_XXX` may exist in more than one partition.
2. **Hash Deduplication:**
   - Exact match: SHA-256 hash collision checks across all splits.
   - Near-duplicate match: Perceptual hashing (pHash) with Hamming distance $\le 4$. Any flagged pairs are manually inspected and assigned to the same partition.
3. **Cross-Dataset Overlap Limitation:**
   - Document explicitly that because Stanford DDI lacks patient identifiers, potential patient overlap between DDI and external web sources cannot be mathematically disproven, representing an inherent dataset documentation constraint.

---

## 6. Threshold Selection Protocol & Honest Reporting Standards

### A. Threshold Selection on Validation Data Only
- The provisional goal of achieving **85–90% disease sensitivity** is an **optimization target on the validation split**, NOT a guaranteed performance metric or a clinically validated safety claim.
- **Candidate Threshold Selection:**
  - Using only the 15% Validation split, generate the ROC curve and precision-recall curve.
  - Choose the decision threshold $t^*$ that achieves $\ge 85\%$ sensitivity for Class 1 on the validation set while maximizing balanced accuracy.
  - Freeze $t^*$.

### B. Frozen Test Set Evaluation Standards
- When evaluating the frozen test set using $t^*$:
  - Report the complete $2\times2$ confusion matrix: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
  - Report: Sensitivity (Recall), Specificity, Precision, Balanced Accuracy, Macro F1, and PR-AUC.
  - Report **Wilson 95% confidence intervals** for sensitivity and specificity.
  - Zero threshold adjustments or retraining permitted on the test set.

---

## 7. Pre-Training Verification Gate (Checklist)

Before any neural network training begins, the following empirical milestones must be satisfied:

- [ ] **Volunteer Consent Documentation:** All Option A volunteer images collected under approved written/digital informed consent.
- [ ] **Automated QC Execution:** 100% of candidate images pass blur, luminance, and skin segmentation checks.
- [ ] **Partition Manifest Verification:** Split manifest generated and frozen with:
  - Subject ID overlap across splits: **Exactly 0**.
  - Duplicate SHA-256 hash collisions across splits: **Exactly 0**.
- [ ] **Demographic Balance Audit:** FST I–II, III–IV, and V–VI present in balanced ratios ($\ge 20\%$ per stratum) in both Train and Test splits.
- [ ] **Class Imbalance Calculation:** Compute exact ratio $N_{\text{disease}} / N_{\text{normal}}$ on the Train split to establish capped class weighting ($\le 5.0\times$) prior to compilation.

---

## 8. Stage B Sign-off & Stop

- **Artifact Created:** Saved to [`reports/normal_disease/stage_b_label_mapping_split_plan.md`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/reports/normal_disease/stage_b_label_mapping_split_plan.md).
- **Compliance:** Zero datasets downloaded, zero splits created, zero models trained, zero packages installed, and zero existing project files altered.
- **Stage Status:** **Stage B plan fully finalized and revised.** Awaiting your explicit approval before proceeding to Stage C.
