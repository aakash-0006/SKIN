# 📱 SkinDiseaseAI — Smartphone Generalization & Input Validation Plan

**Project:** SkinDiseaseAI — Complete Model Accuracy Improvement and Generalization Project  
**Author:** Senior Machine Learning Engineer & Computer Vision Researcher  
**Date:** October 29, 2026  
**Document:** `reports/model_improvement/smartphone_generalization_plan.md`  
**Focus:** Bridging the Dermoscopy-to-Smartphone Domain Gap & Multi-Stage Input Validation  

---

## 1. The Dermoscopy-to-Smartphone Domain Gap

A critical error in consumer dermatology AI is deploying models trained on dermoscopic images directly onto consumer smartphone photographs without accounting for fundamental optical domain shifts:

```
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│       DERMOSCOPIC PHOTOGRAPHY        │     │        SMARTPHONE PHOTOGRAPHY        │
├──────────────────────────────────────┤     ├──────────────────────────────────────┤
│ • Contact plate with liquid immersion│     │ • Non-contact, free-hand capture     │
│ • Uniform polarized/non-polarized LED│     │ • Variable ambient lighting (sun/LED)│
│ • Fixed focal distance & magnification│     │ • Variable focal distance (10–35 cm) │
│ • Peripheral lens vignette ring      │     │ • No vignette; perspective distortion│
│ • Eliminates surface reflection      │     │ • Specular glare & skin fold shadows │
│ • High pigment-structure visibility  │     │ • Subsurface scattering muted        │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

When an EfficientNetB0 feature extractor trained on dermoscopy processes a smartphone image, early and mid-level convolutional filters respond strongly to non-lesion domain artifacts (shadows, flash glare, vellus hair, perspective slant), leading to erratic classifications and catastrophic out-of-distribution behavior.

---

## 2. Audit of the Current Suitability Gate (`models/ood_gate.npz`)

### A. Current Implementation
The current input gate (`src/gate.py`, `models/ood_gate.npz`) executes:
1. File integrity checks (size, readable JPEG/PNG, min side $\ge 64$px).
2. Image quality checks (Laplacian blur variance $\ge 4.32$, mean luminance $\in [50.4, 245]$).
3. Out-of-Distribution (OOD) Mahalanobis distance check:
   $$d_M(x) = \sqrt{(e(x) - \mu)^T \Sigma^{-1} (e(x) - \mu)}$$
   where $e(x) \in \mathbb{R}^{1280}$ is the average-pooled feature embedding from a frozen ImageNet EfficientNetB0, and $\mu, \Sigma$ were fitted on 4,000 dermoscopy training images.
4. **Current Threshold:** `thr = 2673.0803` (chosen on validation dermoscopy images to achieve 97% acceptance).

### B. The Failure Mode on Smartphone Skin
- On dermoscopic lesion images, the Mahalanobis distance typically ranges between $1800 \le d_M \le 2500$.
- On genuine smartphone skin images, the absence of dermatoscope contact plates and the presence of ambient lighting shifts the embedding vector, yielding **$d_M \approx 5755 \gg 2673.08$**.
- **Consequence:** The existing gate **falsely rejects valid smartphone photographs**, rendering smartphone support non-functional while the gate is active in `all_skin` mode.

---

## 3. Recalibration Protocol for Smartphone Domain Support

> [!IMPORTANT]
> **Safety Rule:** `models/ood_gate.npz` is a protected baseline asset and will **NOT** be overwritten. Any new calibration will be saved to a separate versioned artifact: `models/ood_gate_smartphone_v2.npz`.

### A. Dataset Partitioning for Gate Calibration
To prevent threshold overfitting, data must be partitioned into two independent cohorts:
1. **Calibration Split (50%):** Used exclusively to fit the smartphone feature centroid $\mu_{\text{phone}}$, covariance $\Sigma_{\text{phone}}$, and optimize distance threshold $t_M^*$.
2. **Evaluation Split (50%):** Held out completely to measure generalization metrics with 95% Wilson score confidence intervals.

### B. Evaluation Cohorts
- **In-Domain Positive Skin Set:**
  - Real smartphone photos of skin from Option A volunteer cohort ($N \ge 400$ images across FST 1–6).
  - Real clinical smartphone condition images from Google SCIN ($N \ge 400$ images).
- **Out-of-Domain Negative Set ($N = 200$ images):**
  - Unrelated indoor scenes: furniture, walls, office desks, flooring.
  - Personal belongings: clothing fabrics, leather jackets, backpacks, watches.
  - Documents & screens: printed paper, laptop monitors, smartphone displays.
  - Non-human biological: animal fur, pet skin, household plants.
- **Corrupted / Low-Quality Set ($N = 100$ images):**
  - Defocused / motion-blurred skin captures.
  - Extreme specular flash glare ($>15\%$ saturated pixels).
  - Severely underexposed / dark images ($<20$ mean luminance).

### C. Experimental Acceptance Criteria
1. **False Rejection Rate (FRR) on genuine smartphone skin:** $\le \mathbf{5.0\%}$ (at least 95% of valid skin photos must pass).
2. **False Acceptance Rate (FAR) on non-skin objects:** $\le \mathbf{10.0\%}$ (at most 10% of unrelated objects may pass).
3. **FAR on Screen Spoofs / Printed Photos:** $\le \mathbf{5.0\%}$.

---

## 4. Multi-Stage Defense-in-Depth Architecture

To avoid relying solely on a single embedding distance threshold, the production input pipeline implements a **5-stage modular filter**:

```
[ Incoming Image ]
       │
       ▼
[ Stage 1: File & Quality Filter ] ──(Fail)──> [ Reject: Corrupt, Blurry, Dark, Overexposed ]
       │ (Pass)
       ▼
[ Stage 2: Skin-Pixel Color Gate ] ──(Fail)──> [ Reject: Non-Skin / Insufficient Skin Pixels ]
       │ (Pass)
       ▼
[ Stage 3: Anti-Spoofing Screen  ] ──(Fail)──> [ Reject: Screen Moire / Printed Photo Edge ]
       │ (Pass)
       ▼
[ Stage 4: Anatomical / Domain Gate] ──(Fail)──> [ Reject: OOD Background / Unsupported Site ]
       │ (Pass)
       ▼
[ Stage 5: Input Accepted ] ───────────────> [ Forward to Normal vs. Disease Classifier ]
```

### Stage Details:
1. **Stage 1 (Quality Gate):** Laplacian variance $\ge 25.0$, Mean luminance $\in [45, 230]$, saturated pixel fraction $\le 8\%$.
2. **Stage 2 (Skin Color Gate):** Adaptive YCbCr and HSV skin locus thresholding. Requires $\ge 25\%$ skin-colored pixels in the central region of interest.
3. **Stage 3 (Anti-Spoofing Gate):** High-frequency 2D FFT analysis to detect repetitive pixel grids (screen moiré) and Canny edge straight-line detection to reject photographs of printed paper frames.
4. **Stage 4 (Domain Gate):** Recalibrated Mahalanobis distance ($d_M \le t_{\text{phone}}^*$) on smartphone feature representations.
5. **Stage 5 (Abstention Policy):** If any stage fails, the pipeline cleanly **abstains** and displays an informative user guidance prompt without computing or revealing disease classifications.

---

## 5. Implementation Roadmap

1. **Phase 1 (Readiness & Sourcing):** Collect Option A volunteer smartphone images and curate the 200-image non-skin negative benchmark.
2. **Phase 2 (Fitting):** Extract embeddings via EfficientNetB0, fit $\mu_{\text{phone}}$ and $\Sigma_{\text{phone}}$, sweep thresholds to satisfy FRR $\le 5\%$ and FAR $\le 10\%$.
3. **Phase 3 (Validation):** Evaluate once on held-out evaluation split and save to `models/ood_gate_smartphone_v2.npz`.
4. **Phase 4 (Streamlit Integration):** Connect the validated smartphone gate into the Streamlit UI behind the "Smartphone Photo Screening" mode selector.
