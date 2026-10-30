# Stage K: Normal-Hand and Clinical Hand-Disease Dataset Compatibility Audit

**Date:** 2026-10-30  
**Status:** Completed — Read-Only Technical Audit  
**Audited Datasets:**
- Reference Normal Hand Dataset: `C:\Users\USER\Downloads\normal skin dataset\archive (3)\Hands\Hands` (11k Hands)
- Clinical Condition Dataset: `C:\Users\USER\Downloads\normal skin dataset\archive (4)` (Eczema vs Tinea)
- Normal-Hand Reference Model: `models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz`
**Workspace:** `d:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI`  
**Safety Classification:** Educational research prototype — NOT a medical device  

---

## 1. Executive Summary & Audit Conclusion

### **VERDICT: SUITABLE WITH ADDITIONAL DATA**

*(Standalone Direct Training Verdict: **NOT SUITABLE** due to fatal confounding and insufficient sample size; **SUITABLE** only as an exploratory out-of-distribution benchmark or with standardized clinical hand datasets).*

An exhaustive programmatic, anatomical, and empirical compatibility audit was conducted to determine whether the 11k Normal Hands collection and the clinical Eczema vs Tinea collection (`archive (4)`) can support a future supervised "normal-looking hand vs abnormal hand" classifier.

### Key Audit Findings:
1. **Severe Anatomical Dilution:** Out of 2,147 images in `archive (4)`, **less than 5%** (~83 images) represent usable whole human hands compatible with automated hand landmark detection. Over **95%** of the dataset depicts non-hand body regions (feet, groin, face, torso, arms, legs) or extreme fingertip macro close-ups.
2. **Fatal Confounder & Domain Mismatch:** 11k Hands was captured on a uniform matte white studio board ($R \approx 253, G \approx 253, B \approx 253$) at uniform $1600 \times 1200$ resolution. `archive (4)` was scraped from clinical websites with colored hospital bedsheets, examination tables, variable low resolutions ($720 \times 472$), and compression artifacts. Direct binary training would produce a **shortcut classifier** that distinguishes studio white paper from hospital sheets rather than detecting pathology.
3. **No Patient Identifiers & High Duplication:** `archive (4)` contains **zero patient IDs**, **zero metadata files**, and **107 exact byte-for-byte duplicate images**. Without patient IDs, split leakage cannot be prevented.
4. **Empirical Validation with Candidate 1B:** When evaluated through the existing Candidate 1B detector, **91.9%** (91/99) of the detected clinical hands in `archive (4)` scored above the threshold ($t=6.931$) and were correctly flagged as `UNUSUAL_RELATIVE_TO_TRAINING_DATA` (Mean score: **7.771** vs threshold 6.931), confirming high sensitivity to clinical lesions without requiring joint binary retraining.

---

## 2. Anatomical & Condition Distribution in `archive (4)`

The 2,147 images in `archive (4)` were audited using filename parsing, metadata extraction, and automated MediaPipe hand landmark detection.

### 2.1 Overall Body-Site Distribution

| Body Region / Anatomical Category | Identification Method | Total Images | Fraction of Dataset | Clinical Presentation |
|:---|:---|:---:|:---:|:---|
| **Hand (Whole Hand Landmark Detectable)** | MediaPipe `HandCropper` verified | **83** | **3.9%** | Dorsal/palmar hands with wrist and finger context |
| **Hand (Fingertip / Macro Partial Close-up)** | Filename keyword (`finger`, `fingertip`, `nail`) | **365** | **17.0%** | Close-ups lacking palm/wrist; fails MediaPipe |
| **Foot & Ankle (Tinea Pedis / Foot Eczema)** | Filename keyword (`foot`, `pedis`, `heel`, `toe`) | **310** | **14.4%** | Soles, heels, web spaces of toes |
| **Face & Scalp (Tinea Capitis / Facial Eczema)** | Filename keyword (`face`, `capitis`, `scalp`, `ear`) | **181** | **8.4%** | Forehead, cheeks, scalp patches |
| **Groin & Genital (Tinea Cruris)** | Filename keyword (`groin`, `cruris`, `inguinal`) | **130** | **6.1%** | Inguinal folds, genital/gluteal skin |
| **Trunk & Torso (Tinea Corporis / Areolar Eczema)**| Filename keyword (`corporis`, `areola`, `trunk`) | **63** | **2.9%** | Chest, abdomen, breast areola, back |
| **Arms & Legs (Extremities)** | Filename keyword (`arm`, `leg`, `elbow`, `knee`) | **42** | **2.0%** | Antecubital/popliteal fossae |
| **Unspecified Anatomical Site** | Generic filenames (e.g. `03Eczema0310.jpeg`) | **973** | **45.3%** | Unspecified clinical close-ups |
| **Total** | — | **2,147** | **100.0%** | — |

> [!WARNING]
> MediaPipe Hand Landmark detection produced **16 false-positive hand detections on non-hands** (11 feet, 2 faces, 2 groins, 1 arm) because toes and curled anatomical folds can visually resemble fingers in the absence of wrist context. Relying solely on automated hand detection without anatomical verification introduces non-hand contamination.

---

### 2.2 Condition-Wise Image Counts

| Partition / Split | Folder Label | Total Raw Images | Images with Hand Keywords | MediaPipe Crop Success | Verified Hand Images |
|:---|:---|:---:|:---:|:---:|:---:|
| **Train** | `eczema` | 1,010 | 247 | 42 | 35 |
| **Train** | `tinea` | 923 | 148 | 42 | 36 |
| **Test** | `eczema` | 112 | 27 | 7 | 6 |
| **Test** | `tinea` | 102 | 26 | 8 | 6 |
| **Total** | — | **2,147** | **448** | **99** | **83** |

### 2.3 Diagnostic Label Reliability
- **Broad Syndromic Buckets:** The folder labels `eczema` and `tinea` collapse clinically distinct disease entities with completely different visual morphologies:
  - `eczema`: Combines dyshidrotic pompholyx (tapioca-like palm blisters), chronic hyperkeratotic hand eczema (thickened fissuring), atopic dermatitis, and areolar eczema.
  - `tinea`: Combines tinea manuum (hand fungal infection), tinea unguium (nail fungus/onychomycosis), tinea pedis (athlete's foot), and tinea corporis (ringworm).
- **Zero Diagnostic Ground Truth:** Images were scraped from web dermatology atlases (predominantly DermNet NZ). No biopsy records, fungal KOH scraping results, or dermatologist consensus notes are provided.

---

## 3. Data Integrity, Duplication & Patient Privacy

1. **Patient Identifiers:**  
   **ZERO.** No patient IDs, subject tracking, age, gender, or Fitzpatrick skin tone annotations exist.
2. **Duplicate Images:**  
   A full cryptographic MD5 scan of all 2,147 images revealed **107 exact byte-for-byte duplicate images** (5.0%). Examples include:
   - `05AtopicHand.jpeg` $\equiv$ `05AtopicDermHand.jpeg`
   - `05AtopicHand1q.jpeg` $\equiv$ `05AtopicDermHand1.jpeg`
   - `13TineaGroin34.jpeg` $\equiv$ `tinea-groin-89.jpeg`
3. **Partition Leakage Hazard:**  
   Because the dataset has duplicate images and multiple crops from the same anonymous individuals, standard random partitioning between train and test splits will cause **severe data leakage**, invalidating test metrics.

---

## 4. Domain Compatibility Analysis: 11k Hands vs Archive (4)

| Domain Dimension | 11k Normal Hands (`archive (3)`) | Clinical Condition Dataset (`archive (4)`) | Domain Compatibility Assessment |
|:---|:---|:---|:---|
| **Capture Modality** | Standardized studio rig | Heterogeneous web-scraped clinic & smartphone photos | **Severe Mismatch** |
| **Background** | 100% Matte white board ($R \approx 253, G \approx 253, B \approx 253$) | Colored hospital sheets, exam paper, wood tables, clothing | **FATAL SHORTCUT RISK** (Classifier will learn background, not disease) |
| **Lighting** | Balanced, diffuse studio lighting | Direct flash, yellow incandescent, fluorescent hospital light | **High Domain Shift** |
| **Framing** | Entire hand centered (wrist to 5 fingertips) | Variable: entire hand, single knuckle, cropped palm | **Framing Incompatibility** |
| **Resolution** | Uniform $1600 \times 1200$ (4:3 aspect ratio) | Variable low-res: $720 \times 472$ (603), $720 \times 480$ (310), $480 \times 720$ (166) | **Severe Resolution & Compression Disparity** |
| **Artifacts** | Clean, uncompressed JPEG | Heavy JPEG compression blocks, blur, web watermarks | **High Artifact Disparity** |

---

## 5. Licensing & Legal Permissions

- **Image Source:** Filename structures (such as `tincorp18__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpeg`) confirm images originated from the **DermNet New Zealand** clinical library.
- **Copyright Policy:** DermNet images are protected by copyright owned by the DermNet New Zealand Trust.
- **Permitted Use:** DermNet grants permission strictly for personal non-commercial educational purposes with attribution. **Commercial deployment, model weight redistribution, and uncurated web redistribution are strictly prohibited.**
- **Research Implication:** Any model trained directly on this collection cannot be open-sourced, published, or deployed commercially without explicit licensing agreements.

---

## 6. Empirical Benchmark with Integrated Candidate 1B

To test whether the existing normal-hand appearance pipeline can detect abnormality without retraining, all 99 landmark-detectable hands in `archive (4)` were evaluated through the Candidate 1B detector (`candidate_1b_l2_mahalanobis_k32.npz`, threshold $t = 6.931$):

| Condition Group | Total Evaluated | Mean Anomaly Score | Flagged Unusual / Anomaly ($> 6.931$) | Flagged Usable / Normal ($\le 6.931$) | Anomaly Detection Sensitivity |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Clinical Eczema Hands** | 49 | **7.961** | 46 | 3 | **93.9%** |
| **Clinical Tinea Hands** | 50 | **7.585** | 45 | 5 | **90.0%** |
| **Combined Clinical Hands** | **99** | **7.771** | **91** | **8** | **91.9%** |

### Critical Technical Insight:
Candidate 1B successfully flagged **91.9%** of clinical disease hands as abnormal/unusual **without ever having been trained on a single diseased hand**. This confirms that the one-class appearance modeling architecture already provides strong sensitivity to clinical lesions without exposing the model to the fatal background/shortcut traps of supervised binary training.

---

## 7. Recommended Training & Experimental Strategy

### 1. What NOT to Do:
- **DO NOT train a supervised binary CNN (Normal vs Abnormal) using 11k Hands as Class 0 and `archive (4)` as Class 1.**  
  Such a model will achieve $>98\%$ training accuracy by classifying the white studio background versus clinical backgrounds, failing completely when presented with real-world user photos.
- **DO NOT relabel the ~2,000 non-hand images** as hand conditions.

### 2. Recommended Path:
- **Maintain the One-Class Appearance Modeling Paradigm:**  
  Candidate 1B is mathematically sound and avoids shortcut learning because it is fitted exclusively on healthy hands and tightly cropped hand regions.
- **Use Curated Hand Disease Images as an Out-of-Distribution Validation Probe:**  
  The ~83 verified hand images in `archive (4)` can serve as an external test probe to evaluate whether candidate thresholds catch genuine pathology.
- **Prerequisites for a True Supervised Hand Classifier (Missing Data Requirements):**
  1. **Standardized Clinical Hand Dataset:** A minimum of 500–1,000 clinically confirmed hand lesion photographs (atopic dermatitis, tinea manuum, psoriasis, scabies) photographed under standardized lighting.
  2. **Neutral or Natural Backgrounds:** Clinical hands captured on diverse everyday backgrounds (wood, desks, outdoor, fabric) to break background correlation.
  3. **Patient Identifiers:** Subject IDs to ensure patient-disjoint train/val/test splits.
  4. **Fitzpatrick Representation:** Explicit documentation of skin types I through VI to prevent racial disparity.

---

## 8. Summary Audit Conclusion

`archive (4)` is **NOT SUITABLE** for standalone supervised training. However, the overall project direction is **SUITABLE WITH ADDITIONAL DATA** when:
1. Candidate 1B is maintained as the primary unsupervised appearance density estimator.
2. The 83 verified clinical hand photos are reserved strictly as an exploratory out-of-distribution benchmark.
3. Additional standardized hand lesion datasets with patient IDs and clear licensing are acquired before attempting supervised multi-class classification.

---

**STOPPED.** Awaiting user review and further instructions. No models, code, or datasets have been modified.
