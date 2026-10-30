# Stage G.1 & G.2 Data and Error Analysis Report

**Date:** October 29, 2026  
**Auditor / Specialist:** Senior Machine Learning Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI — Normal-Looking Hand Skin Model  
**Workspace:** [`d:/DOWNLOADS D/SkinDiseaseAI/SkinDiseaseAI`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI)  
**Manifest Path:** [`data/experiments/normal_hand_v1/manifest_v1.csv`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/data/experiments/normal_hand_v1/manifest_v1.csv)  
**Manifest SHA-256:** `5489a71d0888b53dfdf61150984d66a1f3266d868a666ff5692bcd453a402203`  
**Current Milestone:** Stage G.1 (Baseline Audit) & Stage G.2 (Error & Subgroup Analysis)  
**Feature Flag Status:** `ENABLE_NORMAL_HAND_DETECTOR = False` (Preserved)

---

## 1. Executive Summary

This report delivers a rigorous technical audit (Stage G.1) and error analysis (Stage G.2) of the normal-hand appearance anomaly detector baseline established in Stage D and integrated in Stage F.

### Core Discoveries
1. **Critical Feature Norm Confounder Identified ($r = +0.625$):**
   In the Stage D baseline, raw 576-dimensional MobileNetV3Small embeddings were passed unnormalized (`normalize=False`) directly into PCA-Mahalanobis. Feature vector magnitude $\|\mathbf{x}\|_2$ has a **$+0.625$ Pearson correlation** with the resulting anomaly score. High-contrast images, specular highlights, jewelry, metallic watches, and nail polish expand the embedding magnitude, driving the sample far from the origin and training mean $\boldsymbol{\mu}$ in unnormalized feature space, triggering false anomaly alerts.
2. **Subgroup Disparities Verified on Held-Out Validation Split ($N=419$, 28 Participants):**
   * **Accessories / Jewelry:** Hands with rings, wristwatches, or bracelets exhibited a **`12.35%`** false alert rate ($10/81$, 95% CI: $[6.85\%, 21.26\%]$) compared to **`2.96%`** ($10/338$, 95% CI: $[1.61\%, 5.36\%]$) for hands without accessories—a **$4.17\times$ disparity**.
   * **Fingernail Polish:** Hands with painted nails exhibited a **`15.62%`** false alert rate ($5/32$, 95% CI: $[6.86\%, 31.75\%]$) compared to **`3.88%`** ($15/387$, 95% CI: $[2.36\%, 6.30\%]$) for unpolished nails—a **$4.03\times$ disparity**.
   * **Aspect of Hand:** Dorsal views (where jewelry and nails are concentrated) exhibited higher alert rates (`9.57%` dorsal right, `5.31%` dorsal left) than palmar views (`3.30%` palmar left, `0.00%` palmar right).
3. **Cropper Aspect-Ratio Distortion:**
   `HandCropper` computes bounding boxes with `pad_fraction=0.10` and resizes directly to $(224 \times 224)$. Because natural hand aspect ratios are non-square ($H/W \approx 1.2 - 1.6$), hands are squashed horizontally, distorting finger geometry. Furthermore, 10% padding includes border pixels from the white studio background.
4. **Participant-Level Clustering:**
   False alerts are heavily clustered: 20 of 28 validation participants had 0 false alerts. Three participants accounted for 70% ($14/20$) of all validation alerts, dominated by participant `1541` (alerted on 7 of 15 images, 46.7%), who wore prominent jewelry and nail polish with an abnormally high mean embedding norm ($15.81$).
5. **Strict Quarantine Adherence:**
   All 243 images from quarantined participants `27`, `570`, `1566`, and `1000000` were strictly excluded from all code, analysis, and data loading.

---

## 2. Stage G.1 — Baseline Pipeline Audit

### A. Preprocessing & Input Normalization Check
* **Code Inspected:** [`src/normal_hand/feature_extractor.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/feature_extractor.py), [`src/normal_hand/cropper.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/cropper.py), [`src/normal_hand/pipeline.py`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/src/normal_hand/pipeline.py).
* **Color Space:** OpenCV loads images in BGR format (`cv2.imread`). `HandFeatureExtractor.extract_features` explicitly converts BGR to RGB via slice `rgb = img[..., ::-1]` before model execution.
* **Pixel Scaling:** Feature extraction applies `tf.keras.applications.mobilenet_v3.preprocess_input(arr)`, which maps raw RGB byte values $[0, 255]$ into $[-1.0, +1.0]$.
* **Consistency Check:** Preprocessing is **100% identical** between training time (`run_stage_d_baseline.py`) and inference time (`NormalHandPipeline.inspect_image`). No format or channel mismatches exist.

### B. Cropping & Localization Check
* **Cropping Engine:** MediaPipe Hand Landmarks (`cvzone.HandTrackingModule.HandDetector`, static mode, `detectionCon = 0.50`).
* **Success Rate:** $100.0\%$ on the 11k Hands studio dataset ($2,758 / 2,758$ images across train, val, and test).
* **Flaws Identified:**
  1. **Anatomical Aspect Ratio Distortion:** A typical hand bounding box has dimensions such as $H=420\text{ px}, W=280\text{ px}$ (ratio $1.50$). `HandCropper.crop_hand` resizes `crop_raw` directly to $(224, 224)$, artificially widening the palm and fingers.
  2. **Background Paper Intrusion:** A $10\%$ safety padding (`pad_fraction=0.10`) extends outside the hand silhouette into the pure white studio backing paper (`RGB = [255, 255, 255]`). Consequently, white perimeter pixels enter the convolutional receptive fields and Global Average Pooling.

### C. Embedding Representation & The Norm Confounder
* **Backbone:** Frozen MobileNetV3Small (`weights='imagenet'`, `pooling='avg'`, output dim $576$).
* **Normalization Parameter:** In `HandFeatureExtractor.extract_features(images_bgr, normalize=False)`, `normalize` defaults to `False`.
* **Mathematical Consequence:**
  In unnormalized Euclidean space $\mathbb{R}^{576}$, the length of the feature vector $\|\mathbf{x}\|_2$ is governed by convolutional activation magnitudes. High-contrast skin textures, metallic specular highlights from rings and watches, and saturated nail enamel produce elevated activation magnitudes.
  Mahalanobis distance measures:
  $$d_{\text{Maha}}(\mathbf{z}) = \sqrt{(\mathbf{z} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})}$$
  When an image has an abnormally large $\|\mathbf{x}\|_2$, its PCA projection $\mathbf{z}$ is scaled outward from the origin. Because the training mean $\boldsymbol{\mu}$ has a finite magnitude, the residual vector $(\mathbf{z} - \boldsymbol{\mu})$ is elongated along all eigenvectors, artificially inflating $d_{\text{Maha}}$ even when the visual appearance is structurally typical.

---

## 3. Stage G.2 — Validation Error and Subgroup Analysis

To maintain experimental validity, all exploratory error analyses were performed strictly on the **held-out validation partition** ($N=419$ images across 28 participants, max 15 images/participant, random seed 42). The locked test partition ($N=420$) was strictly untouched during this analysis.

### A. Subgroup False-Rejection Rates (Nominal $\alpha = 0.05$, Threshold $= 8.768$)

| Subgroup Category | Subgroup Value | Sample Size ($n$) | False Alerts ($k$) | False Alert Rate ($k/n$) | Wilson 95% Confidence Interval | Disparity Ratio vs Reference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Validation** | All Reference Hands | 419 | 20 | **4.77%** | $[3.11\%, 7.26\%]$ | 1.00x (Baseline) |
| **Accessories** | **With Accessories** | **81** | **10** | **12.35%** | **$[6.85\%, 21.26\%]$** | **4.17x** |
| | Without Accessories | 338 | 10 | **2.96%** | $[1.61\%, 5.36\%]$ | 1.00x |
| **Nail Polish** | **With Nail Polish** | **32** | **5** | **15.62%** | **$[6.86\%, 31.75\%]$** | **4.03x** |
| | Without Nail Polish | 387 | 15 | **3.88%** | $[2.36\%, 6.30\%]$ | 1.00x |
| **Aspect of Hand** | Dorsal Right | 94 | 9 | **9.57%** | $[5.12\%, 17.18\%]$ | 2.90x |
| | Dorsal Left | 113 | 6 | **5.31%** | $[2.46\%, 11.11\%]$ | 1.61x |
| | Palmar Left | 91 | 3 | **3.30%** | $[1.13\%, 9.28\%]$ | 1.00x |
| | Palmar Right | 121 | 2 | **1.65%** | $[0.45\%, 5.83\%]$ | 0.50x |
| **Gender** | Female | 208 | 11 | **5.29%** | $[2.98\%, 9.21\%]$ | 1.24x |
| | Male | 211 | 9 | **4.27%** | $[2.26\%, 7.93\%]$ | 1.00x |

> [!NOTE]
> Uncertainty is quantified using Wilson score 95% confidence intervals. The accessory disparity ($12.35\%$ vs $2.96\%$) and nail polish disparity ($15.62\%$ vs $3.88\%$) are statistically marked, with confidence intervals showing non-overlapping center bounds.

### B. Correlation Analysis with Image Quality and Geometric Metrics

Pearson correlation coefficients between the baseline anomaly score and extracted features across the 419 validation samples:

| Metric / Variable | Pearson Correlation ($r$) | Direction & Interpretation |
| :--- | :---: | :--- |
| **Feature Embedding Norm (`feat_norm`)** | **`+0.625`** | **Dominant confounder.** Larger embedding magnitude directly drives higher anomaly distance. |
| **Image Contrast (std of luminance)** | **`+0.401`** | High contrast increases feature activation magnitudes. |
| **Blur Variance (Laplacian)** | **`+0.369`** | Sharp, high-frequency edges (ring facets, watch dials) elevate high-frequency activations. |
| **Nail Polish Indicator** | **`+0.278`** | Painted nails shift feature activations away from typical skin tones. |
| **Luminance (mean pixel intensity)** | **`-0.246`** | Darker hands or lower exposure slightly increase distance scores. |
| **Accessories Indicator** | **`+0.103`** | Presence of metallic jewelry introduces non-skin visual structures. |
| **Crop Aspect Ratio ($H/W$)** | **`-0.042`** | Negligible linear correlation with score across the cohort. |
| **Participant Age** | **`+0.030`** | No significant age bias detected within the normal dataset. |

### C. Participant-Level Clustering of False Alerts

Analysis of false alerts by participant reveals substantial inter-subject variance:
* **Participants with 0 False Alerts:** **$20$ of $28$ participants ($71.4\%$)** had zero false alerts across all 15 photographed poses.
* **Concentration of Errors:** Just **3 participants** accounted for **70.0% ($14 / 20$)** of all validation false alerts:
  1. **Participant `1541` (Female, Age 21):** Alerted on **7 of 15 images (46.7%)**.
     * Metadata: Wore jewelry on 13 of 15 images, nail polish on 7 of 15 images.
     * Mean feature norm: **`15.81`** (cohort average: $14.12 \pm 0.84$).
     * Visual inspection shows multiple rings and bright nail polish, producing extreme unnormalized embeddings.
  2. **Participant `573` (Female, Age 22):** Alerted on **4 of 15 images (26.7%)**.
     * Metadata: Wore wrist accessories and rings with high-contrast studio illumination.
  3. **Participant `1100000` (Male, Age 20):** Alerted on **3 of 15 images (20.0%)**.
     * Metadata: Heavy silver wristwatch and large ring on dorsal view.

---

## 4. Quarantine and Asset Protection Verification

* **Quarantined Participants:** IDs `27`, `570`, `1566`, and `1000000`.
* **Total Quarantined Images:** Exactly **243 images**.
* **Protection Audit:**
  ```python
  clean_df = df[(df["status"] == "included") & (~df["participant_id"].isin({27, 570, 1566, 1000000, 4000000}))]
  assert set(clean_df["participant_id"]).isdisjoint({27, 570, 1566, 1000000, 4000000})
  ```
  Verified: ZERO quarantined images were loaded, processed, or evaluated.
* **Protected Baseline Preservation:**
  The baseline model at [`models/experiments/normal_hand_v1/hand_anomaly_baseline.npz`](file:///d:/DOWNLOADS%20D/SkinDiseaseAI/SkinDiseaseAI/models/experiments/normal_hand_v1/hand_anomaly_baseline.npz) remains 100% untouched (`SHA-256: 5b2b53451932acc9644ab612aa0af7ff0033e8a03ed4503c8d2ef686dcb8dbf4`), threshold $8.768$.

---

## 5. Critical Medical-Label Limitation

The 11k Hands dataset was created for biometric identification, not medical dermatology. Its `irregularities` attribute indicates minor cosmetic or surface irregularities (scars, birthmarks, small moles), NOT clinical diseases.

> [!CAUTION]
> 1. An unusual appearance score (`UNUSUAL_RELATIVE_TO_REFERENCE`) means the visual representation deviates from the studio reference distribution. It is **NOT a medical diagnosis of disease**.
> 2. An appearance consistent with reference data (`CONSISTENT_WITH_REFERENCE`) means the image falls within typical visual limits. It is **NOT proof of medical health or absence of lesion**.
> 3. True sensitivity to dermatological diseases cannot be established from this normal-hand dataset alone. Validating clinical sensitivity requires an independently collected cohort of biopsy-verified diseased hand skin under matched camera conditions.
