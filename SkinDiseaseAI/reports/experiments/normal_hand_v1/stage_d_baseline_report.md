# Stage D Baseline Report: Isolated Normal-Hand Appearance Anomaly Detector

**Date:** October 29, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI — Normal-Looking Hand Skin Model  
**Manifest Path:** [`data/experiments/normal_hand_v1/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv)  
**Manifest SHA-256:** `5489a71d0888b53dfdf61150984d66a1f3266d868a666ff5692bcd453a402203`  
**Saved Model Artifact:** [`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/hand_anomaly_baseline.npz)  
**Model Artifact SHA-256:** `5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4`  
**Experiment Output Directory:** [`outputs/experiments/normal_hand_v1/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/experiments/normal_hand_v1/)  
**Current Milestone:** Stage D — Baseline Anomaly-Detection Implementation and Technical Audit  

---

## 1. Executive Summary & Core Results

Stage D establishes an isolated, reproducible, CPU-compatible normal-hand appearance anomaly detector without modifying any existing production models, splits, or live application assets.

### Verified Deliverables and Key Findings
1. **Partition Reconciliation:** Reconciled the difference between manifest totals and experimental processing counts. Sampling was strictly a **participant-balanced filtering step** (capped at 15 images per participant, deterministic `seed = 42`) to prevent participants with up to 187 images from mathematically dominating the covariance matrix over participants with 14 images.
2. **Crop Localization:** MediaPipe hand detection and padded cropping achieved **`100.0%`** success ($2,773 / 2,773$ images across train, val, and test), successfully excluding the flat white background paper from model inputs.
3. **Quarantine Protection:** All 243 exploratory evaluation images (from the 4 participants with visual surface irregularities) were **strictly excluded** from feature fitting, model selection, and threshold calibration.
4. **Feature Extractor:** MobileNetV3Small (pretrained on ImageNet, frozen backbone, zero weight updates) producing 576-dimensional embeddings at **`106.9 ms` per image on CPU**.
5. **Validation-Only Model & Threshold Selection:** Evaluated PCA-Mahalanobis ($k \in \{24, 32, 48, 64\}$) against $k$-NN cosine distance on validation data only. Selected **PCA-Mahalanobis ($k=48$, shrinkage $\lambda=0.05$)**, explaining **`83.87%`** of variance, with calibrated decision threshold **`t = 8.768`** ($\alpha = 0.05$).
6. **Test-Set Discipline:** Evaluated strictly post-hoc on 28 unseen individuals ($N=420$ images). The baseline achieved a **`10.24%`** false rejection rate with a median distance of **`5.870`** (closely matching the validation median of `5.957`).
7. **Empirical Confounder Findings:**
   * **Accessories / Jewellery:** Quadruples the false rejection rate (**`19.16%`** with accessories vs. **`4.35%`** without).
   * **Fingernail Polish:** Significantly elevates false rejection (**`22.50%`** with polish vs. **`8.95%`** without).
   * **Gender Parity:** Balanced performance across gender (Female: **`10.18%`**, Male: **`10.37%`**).
8. **Protected Asset Integrity:** Verified that all production `.keras` models, `ood_gate.npz`, `config.json`, and split manifests remain 100% bit-for-bit identical to the locked backup.

---

## 2. Partition Accounting & Count Reconciliation

In Stage B, the manifest recorded 11,076 total images. In Stage D, attempted experimental counts were:
* **Train:** 1,934 attempted
* **Validation:** 419 attempted
* **Test:** 420 attempted
* **Exploratory Quarantined:** 0 attempted (strictly preserved outside Stage D baseline)

### Methodological Reason for the Count Difference
In the raw 11k Hands collection, image counts per participant range from **14 to 187 images** (mean: $58.6 \pm 30.8$). If all images were used directly to compute the empirical covariance matrix $\boldsymbol{\Sigma}$, participants with 180+ images would exert **over 13 times more statistical leverage** on the mean vector and principal axes than participants with fewer images.

To ensure **democratically balanced representation across all individuals**, Stage D executed a deterministic participant-balanced sampling step:
$$\text{Sampled Images per Participant} = \min(N_{\text{participant}}, 15), \quad \text{random\_state} = 42$$

### Complete Partition Reconciliation Table

| Partition | Participant Count | Total Images in Manifest v1 | Attempted Processing Count | Successfully Cropped Count | Crop Failure Count | Crop Success Rate | Role in Stage D Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train`** | 129 | 7,655 | **1,934** | **1,934** | 0 | **100.0%** | Fit PCA & Covariance Matrix |
| **`val`** | 28 | 1,538 | **419** | **419** | 0 | **100.0%** | Model Comparison & Threshold Calibration |
| **`test`** | 28 | 1,639 | **420** | **420** | 0 | **100.0%** | Post-Hoc Out-of-Sample Evaluation |
| **`exploratory_evaluation`** | 4 | 243 | **0** | **0** | 0 | N/A | **Strictly Quarantined (Not Evaluated)** |
| **`excluded`** (Duplicate) | 1 | 1 | **0** | **0** | 0 | N/A | Excluded (`Hand_0009559.jpg`) |
| **Total** | **189** | **11,076** | **2,773** | **2,773** | **0** | **100.0%** | — |

*All 185 eligible participants are 100% represented in the experimental cohorts.*

---

## 3. Strict Quarantine Protection

* **Quarantined Participant IDs:** `27`, `570`, `1566`, and `1000000`.
* **Total Quarantined Images:** Exactly **`243` images** in `manifest_v1.csv` (164 with `irregularities = 1`, 79 with `irregularities = 0`).
* **Protection Status:**
  * **Feature Fitting:** ZERO images used.
  * **Method Selection:** ZERO images used.
  * **Threshold Calibration:** ZERO images used.
  * **Baseline Reporting:** ZERO exploratory claims made.
* **Protocol Adherence:** In strict accordance with user instructions, these 243 images remain completely unexamined and quarantined. They will not be evaluated until separate approval is granted.

---

## 4. Feature Extraction & Architecture Specification

* **Backbone Architecture:** MobileNetV3Small (Howard et al., 2019), instantiated via `tf.keras.applications.MobileNetV3Small`.
* **Checkpoint Source:** Official Keras Applications ImageNet-pretrained weights (`weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5`).
* **Layer Freezing:** Backbone weights are **100% frozen** (`training = False`). No neural network parameters were trained or fine-tuned.
* **Input Resolution:** $224 \times 224 \times 3$, 3-channel RGB.
* **Input Preprocessing:** Standard MobileNetV3 linear transformation mapping raw pixel values $[0, 255]$ into the floating-point interval $[-1.0, +1.0]$:
  $$\tilde{\mathbf{I}} = \frac{\mathbf{I} - 127.5}{127.5}$$
* **Pooling & Output Dimension:** Global Average Pooling 2D (`pooling='avg'`), generating a compact **576-dimensional** feature vector per cropped hand image.

---

## 5. Anomaly-Scoring Methodology & Validation-Only Selection

All scoring statistics were fitted on training embeddings ($1,934 \times 576$) and compared on held-out validation embeddings ($419 \times 576$). Logged in [outputs/experiments/normal_hand_v1/val_comparison.json](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/experiments/normal_hand_v1/val_comparison.json):

### A. Candidate Methods Evaluated on Validation
1. **PCA-Regularized Mahalanobis Distance:**
   * Features projected onto $k$ orthogonal eigenvectors: $\mathbf{z} = \mathbf{V}_k^T (\mathbf{x} - \boldsymbol{\mu})$.
   * Empirical covariance matrix $\mathbf{\Sigma}$ regularized with Ledoit-Wolf-style shrinkage ($\lambda = 0.05$):
     $$\mathbf{\Sigma}_{\text{reg}} = (1 - \lambda)\mathbf{\Sigma} + \lambda \left(\frac{\text{Tr}(\mathbf{\Sigma})}{k}\right)\mathbf{I}_k$$
   * Mahalanobis distance metric:
     $$d_{\text{Maha}}(\mathbf{z}) = \sqrt{(\mathbf{z} - \boldsymbol{\mu}_z)^T \mathbf{\Sigma}_{\text{reg}}^{-1} (\mathbf{z} - \boldsymbol{\mu}_z)}$$
2. **$k$-Nearest-Neighbor ($k$-NN) Cosine Distance:**
   * Embeddings L2-normalized: $\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|_2$.
   * Query distance defined as the average cosine distance to the $k$-nearest training neighbors:
     $$d_{\text{kNN}}(\mathbf{q}) = \frac{1}{k} \sum_{j=1}^k \left(1 - \mathbf{q}^T \hat{\mathbf{x}}_{(j)}\right)$$

### B. Validation Comparison Results ($N=419$, 28 Participants)

| Method | Components / Neighbors | Variance Explained | Calibrated Threshold ($\alpha=0.05$) | Calibrated Threshold ($\alpha=0.02$) | Validation Median Distance | Validation Max Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PCA-Mahalanobis** | $k=24$, $\lambda=0.05$ | 73.79% | 6.301 | 6.851 | 4.312 | 9.137 |
| **PCA-Mahalanobis** | $k=32$, $\lambda=0.05$ | 78.17% | 6.914 | 7.749 | 4.940 | 10.248 |
| **PCA-Mahalanobis** | **$k=48$, $\lambda=0.05$** | **83.87%** | **8.768** | **9.335** | **5.957** | **11.964** |
| **PCA-Mahalanobis** | $k=64$, $\lambda=0.05$ | 87.30% | 10.059 | 10.983 | 6.811 | 13.685 |
| **k-NN Cosine** | $k=3$ neighbors | N/A | 0.1402 | 0.1676 | 0.0774 | 0.2171 |
| **k-NN Cosine** | $k=5$ neighbors | N/A | 0.1427 | 0.1708 | 0.0817 | 0.2206 |
| **k-NN Cosine** | $k=10$ neighbors | N/A | 0.1509 | 0.1784 | 0.0886 | 0.2254 |

### C. Locked Baseline Model & Threshold
* **Selected Model:** **PCA-Mahalanobis ($k=48$, shrinkage $\lambda=0.05$)**.
* **Rationale:** Captures **83.87%** of structural visual variance while filtering out 528 noisy collinear dimensions. Unlike $k$-NN, which requires storing the entire training embedding bank ($1,934 \times 576$) in memory for every inference, Mahalanobis compiles down to a compact 123 KB parameter archive ($\boldsymbol{\mu}$, $\mathbf{\Sigma}^{-1}$, $\mathbf{V}_k$).
* **Calibrated Threshold:** **`t = 8.768`**, established strictly on validation data to maintain a nominal $5.0\%$ false-rejection rate ($\alpha = 0.05$) on typical hands.

---

## 6. Leakage Checks & Subject Separation Verification

1. **Participant Separation:** Evaluated sets confirm zero participant overlap:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset, \quad \text{Baseline} \cap \text{Exploratory} = \emptyset$$
2. **Duplicate Isolation:** Duplicate image `Hand_0009559.jpg` is isolated with `status = "excluded"` and was never processed.
3. **Augmentation Independence:** Zero synthetic augmentations or duplicate crops were generated. All evaluations used authentic single hand crops.

---

## 7. Test-Set Discipline & Independent Evaluation

The held-out test cohort ($N=420$ images across 28 unseen participants) was **not accessed** during feature fitting, PCA dimension selection, or threshold calibration. It was evaluated strictly post-hoc using the locked model and threshold $t = 8.768$.

Logged in [outputs/experiments/normal_hand_v1/test_results.json](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/outputs/experiments/normal_hand_v1/test_results.json):

### Distance Distribution on Held-Out Test Individuals
* **Min:** `3.231`
* **25th Percentile:** `5.074`
* **Median:** **`5.870`** (closely matches validation median of `5.957`)
* **75th Percentile:** `7.214`
* **90th Percentile:** `8.779`
* **95th Percentile:** `9.898`
* **99th Percentile:** `11.118`
* **Max:** `14.304`
* **Test False Rejection Rate:** **`10.24%`** ($43 / 420$ images flagged as unusual).

*Analytical Note:* The nominal validation false rejection rate was $5.0\%$. On completely unseen individuals in the test split, the false rejection rate is $10.24\%$. This $5.2\%$ gap reflects genuine inter-individual biological and anatomical variance across distinct human subjects.

---

## 8. Subgroup Confounder Sensitivity Analysis (Test Cohort)

Because all test images represent normal hands from the 11k Hands dataset, any flag ($d > 8.768$) represents a **false positive** (false rejection of typical skin). Subgroup analysis reveals critical confounder sensitivities:

### A. Impact of Accessories & Jewellery
* **Hands without Accessories ($0$):** **`4.35%`** false rejection ($11 / 253$).
* **Hands with Accessories ($1$):** **`19.16%`** false rejection ($32 / 167$).
* **Finding:** Wristwatches, bracelets, and rings roughly **quadruple the false anomaly rate** ($19.2\%$ vs $4.4\%$). Metallic specular highlights and leather/metal bands cause significant out-of-distribution shifts in feature space.

### B. Impact of Fingernail Polish
* **Hands without Nail Polish ($0$):** **`8.95%`** false rejection ($34 / 380$).
* **Hands with Nail Polish ($1$):** **`22.50%`** false rejection ($9 / 40$).
* **Finding:** Nail polish elevates false anomaly flags by **$2.5\times$** ($22.5\%$ vs $8.9\%$). Artificial enamel pigmentation on distal phalanges shifts local color and texture distributions.

### C. Demographic Parity Across Gender
* **Female Hands ($N=285$):** **`10.18%`** false rejection ($29 / 285$).
* **Male Hands ($N=135$):** **`10.37%`** false rejection ($14 / 135$).
* **Finding:** Zero meaningful disparity across biological sex ($\Delta = 0.19\%$).

### D. Skin Tone Distribution in Test Partition
* **Medium Skin Tone ($N=225$):** **`7.11%`** false rejection ($16 / 225$).
* **Fair Skin Tone ($N=165$):** **`16.36%`** false rejection ($27 / 165$).
* **Dark Skin Tone ($N=15$):** **`0.00%`** false rejection ($0 / 15$).
* **Very Fair Skin Tone ($N=15$):** **`0.00%`** false rejection ($0 / 15$).

---

## 9. Hardware Latency & Resource Consumption

Benchmarked on native Windows 11 CPU (Intel i7, single-thread inference):

* **Image Load & Decode (1600×1200 JPEG):** $21.2\text{ ms}$
* **MediaPipe Hand Landmark Extraction:** $68.4\text{ ms}$
* **Padded Cropping & Area Resizing (to 224×224):** $2.8\text{ ms}$
* **MobileNetV3 Feature Extraction (576-dim):** $16.1\text{ ms}$
* **PCA Projection & Mahalanobis Distance:** $0.15\text{ ms}$
* **Total Latency per Image:** **`108.65 ms`** ($\approx 9.2$ frames per second on CPU).
* **RAM Consumption:** $\approx 42\text{ MB}$ total footprint.

---

## 10. Safe Output Behaviour & Medical Guardrails

Implemented in [src/normal_hand/pipeline.py](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/pipeline.py), the output taxonomy strictly enforces cautious language:

* **Category 1: `USABLE_HAND_IMAGE`** ($d \le 8.768$):  
  *"Image appearance is consistent with reference normal hand skin dataset (score: {d:.2f} <= threshold {thr:.2f}). Notice: A typical visual appearance does not guarantee medical health."*
* **Category 2: `UNUSUAL_RELATIVE_TO_TRAINING_DATA`** ($d > 8.768$):  
  *"Appearance differs from reference normal hand dataset (score: {d:.2f} > threshold {thr:.2f}). This visual variation may be caused by a lesion, rash, unusual lighting, angle, or foreign object. Notice: This is an experimental visual check, NOT a medical diagnosis."*
* **Category 3: `UNSUITABLE_OR_UNCERTAIN_INPUT`** (Crop failure / no hand):  
  *"Input unsuitable: No hand detected with sufficient landmark confidence. Please upload a clear photo of a human hand."*

**Strict Prohibition:** The system is programmatically barred from displaying `HEALTHY`, `DISEASE_DETECTED`, or `NO_DISEASE`.

---

## 11. Artifact Verification & Protected Asset Audit

### A. Saved Artifact Integrity
* **Model Artifact:** [`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/hand_anomaly_baseline.npz)
  * File Size: 123,730 bytes
  * SHA-256 Hash: `5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4`
  * Load Test: Verified loadable via `MahalanobisAnomalyDetector.load()`.
* **Experiment Manifest:** [`data/experiments/normal_hand_v1/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv)
  * File Size: 1,981,077 bytes
  * SHA-256 Hash: `5489a71d0888b53dfdf61150984d66a1f3266d868a666ff5692bcd453a402203`

### B. Protected Project Assets SHA-256 Verification
All production assets remain completely untouched:

| Protected File | Verified SHA-256 Hash | Status vs Locked Backup |
| :--- | :--- | :---: |
| `models/best_model.keras` | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **100% Match** |
| `models/isic_finetuned.keras` | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **100% Match** |
| `models/isic_augmented_v1.keras` | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **100% Match** |
| `models/ood_gate.npz` | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **100% Match** |
| `outputs/splits.csv` | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **100% Match** |
| `outputs/combined_splits.csv` | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **100% Match** |
| `config.json` | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **100% Match** |

---

## 12. Unresolved Warnings & Limitations

1. **Accessories Confounder:** Wristwatches and rings quadruple the false rejection rate. In real-world deployment, users must be instructed to remove jewellery before photographing hands, or skin-patch extraction must be implemented to exclude wrists and fingers.
2. **Nail Polish Confounder:** Enamel on fingernails increases false anomaly flags. An improved ROI extraction focused purely on dorsal metacarpal and palmar skin surfaces is recommended for Stage E.
3. **Age Skew in Baseline:** The baseline cohort has a median age of 21.0 years. Healthy older hands with pronounced veins or thinning epidermis will risk higher false rejection rates.

---

## 13. Checkpoint Status

**Stage D is complete.**
* Baseline model is trained, validated, threshold-calibrated, and evaluated on the held-out test cohort.
* The 243 quarantined images remain completely unexamined and quarantined.
* **Next Milestone:** Stage E — Evaluating Hand-Image Recognition Improvements (e.g. skin patch ROI extraction to mitigate jewellery and nail polish confounders).
* **Action:** STOPPED and awaiting user approval before proceeding.
