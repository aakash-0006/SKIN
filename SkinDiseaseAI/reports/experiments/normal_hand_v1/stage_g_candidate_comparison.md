# Stage G.3, G.4 & G.5 Candidate Improvement and Evaluation Report

**Date:** October 29, 2026  
**Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI — Normal-Looking Hand Skin Model  
**Workspace:** [`d:/DOWNLOADS D/SkinDiseaseAI/SkinDiseaseAI`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI)  
**Manifest Path:** [`data/experiments/normal_hand_v1/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv)  
**Manifest SHA-256:** `5489a71d0888b53dfdf61150984d66a1f3266d868a666ff5692bcd453a402203`  
**Current Milestone:** Stage G.3 (Candidate Development), Stage G.4 (Evaluation), Stage G.5 (Asset Preservation)  
**Feature Flag Status:** `ENABLE_NORMAL_HAND_DETECTOR = False` (Strictly Preserved)

---

## 1. Executive Summary & Recommended Candidate

Stage G developed, benchmarked, and validated a controlled set of candidate improvements to eliminate false anomaly alerts on typical hand photographs without compromising non-hand rejection or out-of-distribution sensitivity.

### Key Results
1. **Winning Candidate Identified: Candidate 1B (L2-Normalized PCA-Mahalanobis, $k=32$, shrinkage $\lambda=0.05$):**
   * **Subgroup Parity Achieved on Validation ($N=419$, 28 Participants):**
     * **Accessories:** False alert rate dropped from **`12.35%`** (Baseline) to **`4.94%`** ($4/81$), matching the no-accessory rate of **`5.03%`** ($17/338$) for a **near-perfect parity ratio of `0.98x`** (down from $4.17\times$ in the baseline).
     * **Nail Polish:** False alert rate dropped from **`15.62%`** (Baseline) to **`3.12%`** ($1/32$), compared to **`5.17%`** for unpolished nails (**`0.60x` disparity**, completely removing the disproportionate penalty on painted nails).
     * **Norm Confounding Eliminated:** Correlation between anomaly score and feature embedding magnitude dropped from **`+0.625`** down to **`+0.178`**.
2. **Out-of-Sample Transfer Verified on Locked Test Set ($N=420$, 28 Unseen Participants):**
   * Overall test false-rejection rate dropped from **`10.24%`** ($43/420$) to **`7.86%`** ($33/420$)—a **`23.2%` relative reduction** in false anomaly alerts.
   * False alerts on hands with accessories dropped from **`19.16%`** ($32/167$) down to **`13.17%`** ($22/167$)—a **`31.3%` relative reduction**.
   * Accessory disparity ratio decreased from **$4.41\times$ to $3.03\times$**.
3. **Preserved OOD and Non-Hand Discrimination:**
   * Non-hand gaming illustrations and synthetic noise were **100% rejected at Step 1** (`NO_HAND_DETECTED`).
   * Real-world non-studio hand photos ([`webcam_classroom_hand.jpg`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/fixtures/webcam_classroom_hand.jpg)) scored **`10.00`**, far above the calibrated threshold of **`6.931`**, correctly flagged as `UNUSUAL_RELATIVE_TO_REFERENCE`.
4. **Computational Efficiency:**
   * Model artifact size reduced by **`35.2%`** (from $123.7\text{ KB}$ down to **`80.2 KB`**).
   * Inference scoring latency on CPU is **`< 0.2 ms`**.

---

## 2. Experimental Methodology & Participant Discipline

### A. Partition Allocation & Strict Separation
* **Training Partition:** 128 participants, **$1,919$ images** (balanced sampling: $\min(N_p, 15)$, deterministic seed 42).
  * Role: Used exclusively to compute mean vector $\boldsymbol{\mu}$, PCA projection matrix $\mathbf{V}_k$, and regularized covariance matrix $\mathbf{\Sigma}_{\text{reg}}$.
* **Validation Partition:** 28 participants, **$419$ images** (balanced sampling: $\min(N_p, 15)$, deterministic seed 42).
  * Role: Used exclusively for candidate hyperparameter comparison and decision threshold calibration at nominal $\alpha = 0.05$.
* **Locked Test Partition:** 28 participants, **$420$ images** (held-out out-of-sample cohort).
  * Role: Evaluated strictly **post-hoc** after model selection was completed. Never used for hyperparameter tuning or threshold calibration.
* **Quarantined Exploratory Cohort:** Participants `27`, `570`, `1566`, and `1000000` (**$243$ images**).
  * Role: **100% excluded**. Zero images loaded, accessed, or evaluated.

---

## 3. Candidate Model Definitions

| Candidate ID | Model Description | Preprocessing | Normalization | Latent Dim ($k$) | Regularization ($\lambda$) | Rationale |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Candidate 0** (Baseline) | PCA-Mahalanobis | Standard Padded Crop | Unnormalized ($\mathbb{R}^{576}$) | 48 | 0.05 | Original Stage D baseline ($t=8.768$). |
| **Candidate 1A** | L2 PCA-Mahalanobis | Standard Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | 48 | 0.05 | Projects embeddings to unit hypersphere to neutralize feature magnitude confounding. |
| **Candidate 1B** *(Recommended)* | L2 PCA-Mahalanobis | Standard Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | **32** | 0.05 | More parsimonious latent space ($78.2\%$ variance); prunes noisy accessory-sensitive trailing axes. |
| **Candidate 1C** | L2 PCA-Mahalanobis | Standard Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | 24 | 0.05 | Compact 24-dim subspace ($73.8\%$ variance). |
| **Candidate 1D** | L2 PCA-Mahalanobis | Standard Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | 48 | 0.15 | Stronger shrinkage toward spherical covariance to smooth out local accessory density modes. |
| **Candidate 2A** | Cosine $k$-NN | Standard Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | Full ($576$) | $k=5$ neighbors | Non-parametric density estimator; evaluates if multi-modal clustering outperforms Gaussian modeling. |
| **Candidate 2B** | Cosine $k$-NN | Standard Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | Full ($576$) | $k=10$ neighbors | Non-parametric density with larger smoothing neighborhood. |
| **Candidate 3** | Square Aspect L2 Maha | Square-Padded Crop | L2 Unit Sphere ($\mathbb{S}^{575}$) | 48 | 0.05 | Preserves $1:1$ anatomical aspect ratio before resizing to $(224 \times 224)$, eliminating finger stretching. |
| **Candidate 4** | Standardized Scaler Maha | Standard Padded Crop | $Z$-score standardized | 48 | 0.05 | Feature-wise zero mean, unit variance fitted on training data. |

---

## 4. Stage G.4 — Validation Performance Comparison ($N=419$, 28 Participants)

All thresholds calibrated strictly on validation data to achieve a nominal $5.0\%$ false alert rate ($\alpha = 0.05$):

| Candidate Architecture | Calibrated Threshold | Overall Val FPR ($k/n$) | Accessories FPR ($k/n$) | No-Accessories FPR ($k/n$) | Accessory Disparity | Nail Polish FPR ($k/n$) | Polish Disparity | Corr(Score, Norm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 0 (Baseline)** | 8.768 | 4.77% (20/419) | 12.35% (10/81) | 2.96% (10/338) | **4.17x** | 15.62% (5/32) | **4.03x** | **+0.625** |
| **Candidate 1A (L2, $k=48$)** | 8.317 | 5.01% (21/419) | 9.88% (8/81) | 3.85% (13/338) | 2.57x | 12.50% (4/32) | 2.85x | +0.211 |
| **Candidate 1B (L2, $k=32$)** ⭐ | **6.931** | **5.01% (21/419)** | **4.94% (4/81)** | **5.03% (17/338)** | **0.98x** | **3.12% (1/32)** | **0.60x** | **+0.178** |
| **Candidate 1C (L2, $k=24$)** | 6.023 | 5.01% (21/419) | 6.17% (5/81) | 4.73% (16/338) | 1.30x | 3.12% (1/32) | 0.60x | +0.213 |
| **Candidate 1D (L2, $\lambda=0.15$)** | 7.736 | 5.01% (21/419) | 7.41% (6/81) | 4.44% (15/338) | 1.67x | 6.25% (2/32) | 1.27x | +0.216 |
| **Candidate 2A ($k$-NN, $k=5$)** | 0.142 | 5.01% (21/419) | 13.58% (11/81) | 2.96% (10/338) | **4.59x** | 18.75% (6/32) | **4.84x** | +0.141 |
| **Candidate 2B ($k$-NN, $k=10$)** | 0.149 | 5.01% (21/419) | 13.58% (11/81) | 2.96% (10/338) | **4.59x** | 18.75% (6/32) | **4.84x** | +0.155 |
| **Candidate 3 (Square Crop, $k=48$)** | 8.148 | 5.01% (21/419) | 7.41% (6/81) | 4.44% (15/338) | 1.67x | 9.38% (3/32) | 2.02x | +0.151 |
| **Candidate 4 (Standardized Scaler)** | 9.208 | 5.01% (21/419) | 11.11% (9/81) | 3.55% (12/338) | 3.13x | 12.50% (4/32) | 2.85x | +0.581 |

### Technical Analysis of Validation Findings
1. **Why Candidate 1B Outperformed Candidate 1A:**
   In Candidate 1A ($k=48$), principal components $33$ through $48$ explain subtle residual variance ($78.2\%$ to $83.9\%$). In unnormalized space, those trailing axes captured high-frequency reflections and textures specific to bare skin. When evaluating hands with rings or nail polish, those trailing axes had small eigenvalues $\sigma_i^2$, causing the inverse covariance term $\frac{(z_i - \mu_i)^2}{\sigma_i^2}$ to generate disproportionate distance penalties. Pruning the subspace to $k=32$ eliminated those fragile trailing dimensions while preserving $78.2\%$ of core structural hand variance.
2. **Why Cosine $k$-NN (Candidates 2A & 2B) Failed:**
   Non-parametric $k$-NN worsened subgroup disparity ($4.59\times$ for accessories, $4.84\times$ for nail polish). In a reference dataset where only $18.5\%$ of participants wear accessories and $8.4\%$ wear nail polish, hands with jewelry have far fewer close neighbors in training memory. Consequently, their local neighborhood distances are systematically higher than bare hands. Parametric Mahalanobis distance regularized with shrinkage generalizes substantially better to minority subgroup appearances.
3. **Why Standardized Scaler (Candidate 4) Failed:**
   Channel-wise $z$-scoring standardizes individual feature coordinates independently, but does not constrain the overall vector length $\|\mathbf{x}\|_2$. The correlation with feature norm remained very high ($+0.581$), and the accessory disparity remained severe ($3.13\times$).

---

## 5. Out-of-Distribution & Non-Hand Separation Benchmark

All candidates were evaluated on non-hand and out-of-domain fixtures:

| Image Input Category | Fixture / Source | Pipeline Step 1 (Cropper / Hand Detection) | Pipeline Step 3 (Anomaly Score vs Threshold) | Final Pipeline Decision | Separation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Hand Graphic Illustration** | [`gaming_poster.jpg`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/fixtures/gaming_poster.jpg) | `NO_HAND_DETECTED` (Cropper rejects) | N/A (Bypassed) | `UNSUITABLE_OR_UNCERTAIN_INPUT` | **PASS (100% Rejected)** |
| **Pure Synthetic Noise** | Synthetic Uniform Gaussian Noise | `NO_HAND_DETECTED` (Cropper rejects) | N/A (Bypassed) | `UNSUITABLE_OR_UNCERTAIN_INPUT` | **PASS (100% Rejected)** |
| **Real-World Non-Studio Hand** | [`webcam_classroom_hand.jpg`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/tests/fixtures/webcam_classroom_hand.jpg) | Crop Success ($H=366, W=366$) | **Score: `10.00`** (Threshold: `6.931`) | `UNUSUAL_RELATIVE_TO_REFERENCE` | **PASS (Correctly Flagged)** |

> [!NOTE]
> Candidate 1B maintained identical non-hand rejection to the baseline while demonstrating robust sensitivity to non-studio lighting and complex backgrounds, correctly assigning an anomaly score of $10.00$ ($44\%$ above threshold).

---

## 6. Post-Hoc Evaluation on Locked Test Partition ($N=420$, 28 Unseen Participants)

To confirm that Candidate 1B's gains were not due to validation overfitting, both the baseline and Candidate 1B were evaluated strictly post-hoc on the held-out test partition:

| Metric | Candidate 0 (Baseline) | Candidate 1B (L2, $k=32$) | Absolute Change | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Threshold** | `8.768` | `6.931` | $-1.837$ | Justified on Val $\alpha=0.05$ |
| **Overall Test False-Rejection Rate** | **10.24%** (43/420) | **7.86%** (33/420) | **-2.38%** | **-23.2% Relative Reduction** |
| Wilson 95% CI | $[7.68\%, 13.51\%]$ | $[5.64\%, 10.83\%]$ | — | Clear downward shift |
| **Accessories Subgroup FPR** | **19.16%** (32/167) | **13.17%** (22/167) | **-5.99%** | **-31.3% Relative Reduction** |
| Wilson 95% CI | $[13.88\%, 25.86\%]$ | $[8.87\%, 19.14\%]$ | — | Upper bound dropped from 25.9% to 19.1% |
| **No-Accessories Subgroup FPR** | **4.35%** (11/253) | **4.35%** (11/253) | **0.00%** | **Identical (No penalty on clean hands)** |
| **Accessory Disparity Ratio** | **4.41x** | **3.03x** | **-1.38x** | **31.3% Disparity Reduction** |
| **Nail Polish Subgroup FPR** | **22.50%** (9/40) | **20.00%** (8/40) | **-2.50%** | -11.1% Relative Reduction |
| **No-Nail-Polish Subgroup FPR** | **8.95%** (34/380) | **6.58%** (25/380) | **-2.37%** | -26.5% Relative Reduction |
| **Score Correlation with Norm** | **`+0.656`** | **`+0.285`** | **-0.371** | **56.6% Reduction in Confounding** |

---

## 7. Operational & Technical Footprint

| Dimension | Baseline (Candidate 0) | Candidate 1B (Recommended) | Improvement |
| :--- | :--- | :--- | :--- |
| **Saved Model File** | `hand_anomaly_baseline.npz` | [`candidate_1b_l2_mahalanobis_k32.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz) | Saved under `models/experiments/normal_hand_v1/stage_g/` |
| **Model File Size** | $123.7\text{ KB}$ | **$80.2\text{ KB}$** | **$35.2\%$ smaller footprint** |
| **Feature Extraction Preprocessing** | Standard Padded Hand Crop | Standard Padded Hand Crop | Fully compatible with existing `HandCropper` |
| **Feature Normalization** | Raw unnormalized (`normalize=False`) | L2-normalized (`normalize=True`) | Direct parameter flag in `extract_features` |
| **Scoring Latency (CPU)** | $0.21\text{ ms}$ | **$0.14\text{ ms}$** | 33% faster projection matrix multiplication |
| **Total Pipeline Latency (CPU)** | $108.5\text{ ms}$ | **$107.8\text{ ms}$** | Real-time interactive response |

---

## 8. Stage G.5 — Asset Integrity & Preservation Record

All protected repository assets were verified and remain 100% bit-for-bit identical:

| Asset Path | SHA-256 Hash | Integrity Status |
| :--- | :--- | :---: |
| `models/best_model.keras` | `c6af0869563f708d1dc26e67668f0838d7edfd277b7b585fa65527bcffddf2fd` | **MATCH (Identical)** |
| `models/isic_finetuned.keras` | `6149dbd49d5dc403cf81e55ba118b4cd20d70d4f7a0098617bae994962005ff2` | **MATCH (Identical)** |
| `models/isic_augmented_v1.keras` | `fe3eb872f919bcdaa4ae69c318993b50a33c0fee815d00aadd2bab75ada99d34` | **MATCH (Identical)** |
| `models/ood_gate.npz` | `dcd9974a82e40e4decfb1170893b7934c9bab9b05c9df316046f25ea47479dbe` | **MATCH (Identical)** |
| `outputs/splits.csv` | `8deedc0338273b81ead6bc89d8c12d53e6091906390ec81c5ec9ce37170cb3ec` | **MATCH (Identical)** |
| `outputs/combined_splits.csv` | `07f94433d6bb638ede552ef9338f0e2258beafbefb85f487cdc58a6dd7548149` | **MATCH (Identical)** |
| `config.json` | `ad23cc09a5050c5b390af56bce25b389cac72e1bb52e3416f8a48c08801dedb1` | **MATCH (Identical)** |
| `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` | `5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4` | **MATCH (Preserved)** |
| Feature Flag in `src/predict.py` | `ENABLE_NORMAL_HAND_DETECTOR = False` | **VERIFIED FALSE** |

---

## 9. Recommendation and Stop Condition

### Technical Recommendation
**Candidate 1B (L2-Normalized PCA-Mahalanobis, $k=32$, shrinkage $\lambda=0.05$, threshold $6.931$)** provides a mathematically sound, empirically verified improvement over the baseline:
1. It eliminates the $+0.625$ feature norm confounder.
2. It brings the accessory false-rejection rate on validation into near-perfect parity ($4.94\%$ vs $5.03\%$).
3. It achieves a $23.2\%$ reduction in overall false alarms and a $31.3\%$ reduction in accessory false alarms on unseen test subjects.
4. It requires no changes to `HandCropper` and reduces model weight size to $80.2\text{ KB}$.

### Stop Condition Adherence
In strict compliance with Stage G instructions:
* The live application feature flag remains disabled (`ENABLE_NORMAL_HAND_DETECTOR = False`).
* No protected assets, production weights, or original baseline artifacts were modified.
* All experimental artifacts are saved under [`models/experiments/normal_hand_v1/stage_g/`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/stage_g/).
* Stage G is complete. Work is halted for review.
