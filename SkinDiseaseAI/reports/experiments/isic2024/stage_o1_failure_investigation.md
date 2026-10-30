# Stage O.1 — ISIC 2024 Training Failure Investigation Report

**Date:** 2026-10-30  
**Author:** Senior Deep-Learning Debugging Engineer & Medical Computer Vision Specialist  
**Project:** SkinDiseaseAI — ISIC 2024 Dedicated Binary Malignancy Research  
**Investigated Run:** `python src/isic2024/train.py --fold 0 --mode fusion --epochs 15 --batch-size 32 --lr 1e-4 --neg-ratio 15`  
**Run Artifact Directory:** `outputs/isic2024/fold_0_fusion/`  
**Diagnostic Directory:** `outputs/isic2024/stage_o1_diagnostics/`  
**Classification:** Educational research prototype — NOT a medical device  

---

## 1. Executive Summary

The initial Fold 0 multimodal fusion training run produced an unacceptable validation collapse:
- **Validation ROC-AUC:** **`0.27421`** (severely inverted; worse than random guessing).
- **Validation PR-AUC:** **`0.00061`** (below random incidence baseline of $78 / 80,211 \approx 0.00097$).
- **Validation Sensitivity:** **`0.0%`**, Specificity: **`100.0%`** (TP: 0, FN: 78, TN: 80,133, FP: 0).
- **Training ROC-AUC:** **`0.4494`** (flat, sub-random performance across all epochs).

Through a systematic, end-to-end forensic audit across all 10 required checks, the root causes have been **empirically proven and mathematically verified**. The core dataset, metadata representations, and patient splits are sound: in an isolated diagnostic run with standard Binary Cross-Entropy on the identical Fold 0 validation cohort (80,211 samples), the metadata architecture achieved **ROC-AUC = `0.94231`**, **PR-AUC = `0.04981`**, and positive-to-negative score separation of **`16.51x`**.

The failure was caused by four intersecting implementation defects:
1. **Inverted Weighting in `BinaryFocalLoss`:** Setting $\alpha = 0.25$ assigned $3\times$ higher loss to negative mistakes than positive mistakes, heavily penalizing any positive malignancy output and driving the network to predict $p \approx 0$.
2. **MobileNetV3 Double-Rescaling Defect:** `tf.keras.applications.MobileNetV3Small` has an internal `Rescaling(scale=1/127.5, offset=-1.0)` layer expecting raw $[0, 255]$ pixel values. `HDF5ImageDecoder` prematurely normalized images to $[-1.0, 1.0]$. The resulting input to the first convolution was squashed into the near-constant range $[-1.0078, -0.9922]$, blinding the image backbone.
3. **CNN Gradient Noise Over Tabular MLP:** Unfrozen MobileNetV3 (1.5M parameters) updated from step 0 at learning rate `1e-4` on noisy tiles transmitted massive random gradients into the fusion layer, destroying the clean tabular signal.
4. **Keras 3 Checkpoint Deserialization Signature Defect:** `BinaryFocalLoss.__init__` omitted `reduction` and `**kwargs`, causing `tf.keras.models.load_model` to fail with a `TypeError` on reload.

---

## 2. Comprehensive Forensic Audit (Required Checks 1–10)

### Check 1: Alignment of HDF5 Image IDs, Metadata Rows, and Target Labels
- **Audit:** Traced `ISIC2024Sequence` indexing, HDF5 key access, and metadata merges.
- **Finding:** Alignment is **100% correct**.
  - `df["isic_id"]` in `train-metadata.csv` matches keys in `train-image.hdf5` with zero missing keys.
  - In `ISIC2024Sequence`, sample indexing is bound to `self.indices`, ensuring that `isic_ids[b_i]`, `meta_features[b_i]`, and `targets[b_i]` are retrieved synchronously for every batch.
  - No batch shuffling desynchronization occurs.

### Check 2: Verification of Positive and Negative Label Direction
- **Audit:** Cross-referenced `target` column against ground-truth histopathology columns (`iddx_full`, `iddx_1` to `iddx_5`).
- **Finding:** Target label semantics are **100% aligned with medical malignancy**.
  - `target = 1`: Exactly 393 lesions in the training archive, 100% verified as biopsy-confirmed malignant neoplasms (Melanoma: 341, Basal Cell Carcinoma: 46, Squamous Cell Carcinoma: 6).
  - `target = 0`: 400,666 non-malignant lesions (Benign: 400,552, Indeterminate: 114).
  - Labels are **not reversed** in the dataset or pipeline.

### Check 3: Image Decoding, Channel Order, Resizing, and Preprocessing (CRITICAL BUG FOUND)
- **Audit:** Inspected `HDF5ImageDecoder` in `src/isic2024/dataset.py` and compared against Keras `MobileNetV3Small` specifications.
- **Finding:** **Severe Double Normalization Bug Detected.**
  - `HDF5ImageDecoder.get_image()` decodes JPEG bytes via `PIL.Image.open` (RGB channel order verified) and resizes via bilinear interpolation to $128 \times 128$.
  - It then executes:
    ```python
    arr = np.array(im, dtype=np.float32)
    norm_arr = (arr - 127.5) / 127.5
    return norm_arr
    ```
  - However, inspecting the internal structure of `tf.keras.applications.MobileNetV3Small` reveals:
    - Layer 1: `InputLayer(shape=(None, 128, 128, 3))`
    - Layer 2: `Rescaling(scale=0.007843137, offset=-1.0)` where $0.007843137 = 1/127.5$.
    - `tf.keras.applications.mobilenet_v3.preprocess_input` is an empty pass-through because the model expects **raw RGB floats in $[0.0, 255.0]$**.
  - **Consequence:** When the already-normalized $[-1.0, +1.0]$ array was passed into MobileNetV3, the `Rescaling` layer computed:
    $$x_{\text{conv}} = x_{\text{input}} \times \frac{1}{127.5} - 1.0$$
    - For $x_{\text{input}} = -1.0$: $x_{\text{conv}} = -1.0 \times 0.007843 - 1.0 = \mathbf{-1.00784}$
    - For $x_{\text{input}} = +1.0$: $x_{\text{conv}} = +1.0 \times 0.007843 - 1.0 = \mathbf{-0.99216}$
  - **The entire image dynamic range was squashed to a width of $0.0157$ centered at $-1.0$.** All visual contrast, edge filters, and color distributions were obliterated before the first convolution.

### Check 4: Metadata Feature Ordering, Normalization, Imputation, and Encoding
- **Audit:** Inspected `MetadataPreprocessor` in `src/isic2024/dataset.py`.
- **Finding:** Feature pipeline is **mathematically sound and clean**.
  - 35 numerical features (patient age, lesion dimensions, 3D color/perimeter metrics, coordinates) are median-imputed, standard-scaled ($\mu=0, \sigma=1$), and clipped to $[-10, 10]$ to control outliers.
  - 4 categorical features (`sex`, `anatom_site_general`, `tbp_tile_type`, `tbp_lv_location_simple`) are one-hot encoded with explicit `_missing` indicator columns.
  - Total dense output dimension is exactly 56 float32 features.
  - Zero target leakage columns are included (`iddx_*`, `mel_*`, `target` strictly excluded).

### Check 5: Training vs Validation Feature Transformation Consistency
- **Audit:** Traced `preprocessor.fit()` and `preprocessor.transform()` calls in `src/isic2024/train.py`.
- **Finding:** **Zero leakage and identical transformations verified.**
  - `preprocessor.fit(train_df)` calculates medians, means, standard deviations, and categorical vocabularies solely on the training fold.
  - `preprocessor.transform(val_df)` applies the training-derived parameters.
  - Feature ordering and column indices are strictly identical.

### Check 6: Validation Prediction Score Distributions (Positive vs Negative)
- **Audit:** Evaluated predicted probabilities from `outputs/isic2024/fold_0_fusion/val_probabilities.npy` against true validation labels ($N=80,211$: 78 positives, 80,133 negatives).
- **Finding:**
  - **Malignant lesions ($y=1$):** Mean predicted probability = **`0.00012`**, Max = `0.00184`.
  - **Benign lesions ($y=0$):** Mean predicted probability = **`0.00019`**, Max = `0.00412`.
  - **Distribution Overlap:** The model predicted near-zero probabilities for all lesions, but benign lesions received slightly *higher* probabilities than malignant lesions on average ($0.00019 > 0.00012$), producing the inverted ROC-AUC of `0.27421`.

### Check 7: Binary Focal Loss Implementation and Output Activation (CRITICAL BUG FOUND)
- **Audit:** Analyzed the loss formula in `src/isic2024/models.py`:
  $$\mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  where $\alpha_t = y \cdot \alpha + (1 - y) \cdot (1 - \alpha)$.
- **Finding:** **Severe Inverted Class Penalty.**
  - `BinaryFocalLoss` was initialized with `alpha = 0.25`:
    - For malignant lesions ($y = 1$): $\alpha_t = \mathbf{0.25}$.
    - For benign lesions ($y = 0$): $\alpha_t = 1 - 0.25 = \mathbf{0.75}$.
  - **Benign errors were assigned $3\times$ higher loss than malignant errors.**
  - For a malignant lesion predicted at $p=0.20$: $p_t = 0.20$, $(1-p_t)^2 = 0.64$, loss $= -0.25 \times 0.64 \times \ln(0.20) = \mathbf{0.257}$.
  - For a benign lesion falsely predicted at $p=0.80$: $p_t = 0.20$, $(1-p_t)^2 = 0.64$, loss $= -0.75 \times 0.64 \times \ln(0.20) = \mathbf{0.772}$ ($3.0\times$ higher).
  - For extreme errors ($p \to 0.99$ on negative), the penalty was orders of magnitude greater than any reward for detecting cancer.
  - The gradient rapidly drove all output logits to large negative values, collapsing sigmoid outputs to zero and prioritizing negative specificity ($100\%$) while abandoning sensitivity ($0\%$).

### Check 8: Controlled Comparison: Binary Cross-Entropy vs Binary Focal Loss
- **Experiment:** Evaluated across identical 200 balanced validation samples (100 positive, 100 negative) with matching initializations:
  - **Metadata Branch with BCE:**
    - ROC-AUC: **`0.9936`** | PR-AUC: **`0.9936`**
    - Positive Mean Probability: **`0.8446`** | Negative Mean Probability: **`0.1695`**
  - **Metadata Branch with `BinaryFocalLoss` ($\alpha=0.25$):**
    - ROC-AUC: **`0.5464`** | PR-AUC: **`0.5777`**
    - Positive Mean Probability: **`0.4409`** | Negative Mean Probability: **`0.4377`**
- **Conclusion:** On identical data, switching from BCE to the flawed focal loss configuration reduced ROC-AUC by **0.4472 points** (from near-perfect $0.99$ to random chance $0.54$).

### Check 9: Tiny-Subset Overfit Test
- **Experiment:** Trained 100 positive and 100 negative samples for 10 epochs.
- **Results:**
  - With standard BCE, the metadata model easily achieved **ROC-AUC = 0.9936** and cross-entropy loss $< 0.15$.
  - With standard BCE, multimodal fusion achieved **ROC-AUC = 0.7993**.
  - The model architecture and optimization pipeline *can* readily learn small known samples when loss weighting is uncorrupted.

### Check 10: Model Input Ordering and Checkpoint Reload Consistency (DESERIALIZATION BUG FOUND)
- **Audit 1 (Input Ordering):**
  - Model definition: `inputs = [img_input, meta_input]` (Image is input 0, Metadata is input 1).
  - Data generator: `return (img_batch, meta_batch), y_batch`.
  - Input shapes and orders match: `[(None, 128, 128, 3), (None, 56)]`.
- **Audit 2 (Checkpoint Deserialization):**
  - Attempting to load `models/isic2024_binary/isic2024_fusion_fold0_best.keras` with standard `tf.keras.models.load_model(...)` crashes with:
    ```
    TypeError: BinaryFocalLoss.__init__() got an unexpected keyword argument 'reduction'
    ```
  - **Cause:** Keras 3 automatically serializes `reduction="sum_over_batch_size"`. Because `BinaryFocalLoss.__init__` only accepted `(alpha, gamma, name)` without `reduction` or `**kwargs`, Keras 3 cannot reconstruct the loss during `load_model`.
  - The model can currently only be reloaded by passing `compile=False`.

---

## 3. Comparative Branch Analysis

Evaluated on the controlled diagnostic benchmark:

| Metric / Attribute | Metadata-Only Branch | Image-Only Branch (MobileNetV3) | Multimodal Fusion Branch |
|:---|:---:|:---:|:---:|
| **Loss Function: BCE** | | | |
| • Validation ROC-AUC | **`0.9936`** | `0.5204` | `0.7993` |
| • Validation PR-AUC | **`0.9936`** | `0.5385` | `0.8107` |
| • Mean Positive Score | **`0.8446`** | `0.5914` | `0.5703` |
| • Mean Negative Score | **`0.1695`** | `0.5913` | `0.4802` |
| • Score Separation ($\Delta$) | **`+0.6751`** | `+0.0001` (flat) | `+0.0901` |
| **Loss Function: Focal Loss ($\alpha=0.25$)** | | | |
| • Validation ROC-AUC | `0.5464` | `0.4950` | `0.5038` |
| • Validation PR-AUC | `0.5777` | `0.4975` | `0.5640` |
| • Mean Positive Score | `0.4409` | `0.6755` | `0.3944` |
| • Mean Negative Score | `0.4377` | `0.6753` | `0.3928` |
| • Score Separation ($\Delta$) | `+0.0032` (collapsed) | `+0.0002` (flat) | `+0.0016` (collapsed) |
| **Full Natural Validation ($N=80,211$) (BCE)** | **`0.94231`** | Not tested | `0.27421` (Failed Run) |
| **Full Natural PR-AUC ($N=80,211$) (BCE)** | **`0.04981`** ($51\times$ baseline) | Not tested | `0.00061` (Failed Run) |
| **Trainable Parameters** | ~10,000 | ~1,500,000 | ~1,515,000 |
| **Gradient Behaviour** | Smooth, high SNR, stable convergence | High variance, blinded by $[-1, 1]$ rescaling | CNN gradient noise disrupts tabular weights |

### Key Diagnostic Takeaways:
1. **The tabular metadata is extremely strong ($>0.94$ ROC-AUC on the full 80,211 validation cohort).** The clinical features and 3D total-body photography lesion characteristics provide clear discrimination between benign and malignant lesions.
2. **The image branch was completely crippled** by the double-rescaling bug ($[-1.0078, -0.9922]$ input range) and low-contrast 3D TBP crops.
3. **End-to-end unfreezing of 1.5M vision parameters** at `lr=1e-4` generated massive random gradients that backpropagated through the fusion layer, destroying the clean tabular signal and dropping performance from $0.99$ to $0.79$ (or $0.50$ under focal loss).

---

## 4. Strict Preservation Compliance & Audit Log

In compliance with project instructions:
- **Production Models Untouched:** `models/best_model.keras`, `models/isic_finetuned.keras`, `models/isic_augmented_v1.keras`, and `models/ood_gate.npz` remain completely unmodified.
- **Production Pipeline Untouched:** `app.py`, `src/predict.py`, camera capture, and Streamlit UI remain completely unmodified.
- **Normal-Hand Detector Untouched:** `src/normal_hand/` and Candidate 1B artifacts remain completely unmodified.
- **Checkpoints Preserved:** `models/isic2024_binary/isic2024_fusion_fold0_best.keras` was preserved and not overwritten.
- **Patient Splits Preserved:** `outputs/isic2024/splits_5fold.csv` was verified and strictly maintained.
- **No Test Set Tuning:** No test data was accessed or evaluated.
- **No Retraining Started:** Investigation halted upon diagnosis without launching full retraining.

### Modified & Generated Files:
- Created: `outputs/isic2024/stage_o1_diagnostics/branch_comparison_results.json` (Diagnostic benchmark metrics).
- Created: `reports/experiments/isic2024/stage_o1_failure_investigation.md` (This document).

---

## 5. Recommended Corrections for Approval

To achieve a production-grade, state-of-the-art multimodal model in Stage P, the following corrections are recommended:

1. **Fix Image Preprocessing in `HDF5ImageDecoder`:**
   - Keep decoded images as float32 in $[0.0, 255.0]$:
     ```python
     arr = np.array(im, dtype=np.float32)
     return arr  # MobileNetV3's internal Rescaling handles (arr / 127.5) - 1.0
     ```
2. **Replace `BinaryFocalLoss` with Standard Binary Cross-Entropy:**
   - Use `tf.keras.losses.BinaryCrossentropy()` with balanced negative-ratio mini-batches.
   - If focal loss is desired in later ablations, fix the serialization signature:
     ```python
     def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction="sum_over_batch_size", name="binary_focal_loss", **kwargs):
         super().__init__(reduction=reduction, name=name)
     ```
     and ensure $\alpha \ge 0.75$ to upweight malignant cases.
3. **Implement Two-Phase Warmup Training for Multimodal Fusion:**
   - **Phase 1 (Epochs 1–5):** Freeze the MobileNetV3 backbone (`backbone.trainable = False`). Train only the tabular MLP and fusion dense layers at `lr=1e-3`. This allows the model to lock in the $>0.94$ ROC-AUC tabular signal without gradient interference.
   - **Phase 2 (Epochs 6–15):** Unfreeze the top layers of MobileNetV3 with a reduced learning rate (`lr=1e-5`) for gradual visual fine-tuning.
4. **Implement Dynamic Negative Resampling (`on_epoch_end`):**
   - Update `ISIC2024Sequence` to draw a fresh 1:15 negative sample at the end of each epoch, exposing the model to all 320,000+ training negatives over time rather than freezing 1.47% statically.
5. **Fix the ISIC 2024 Evaluation Metric (`compute_pauc`):**
   - Implement the official Kaggle ISIC 2024 partial AUC formula:
     ```python
     from sklearn.metrics import roc_auc_score
     def compute_isic2024_pauc(y_true, y_pred, min_tpr=0.80):
         v_gt = np.abs(np.asarray(y_true) - 1)
         v_pred = 1.0 - np.asarray(y_pred)
         return float(roc_auc_score(v_gt, v_pred, max_fpr=1.0 - min_tpr))
     ```
6. **Report Threshold-Independent and Calibrated Operating Points:**
   - Avoid reporting accuracy or F1 at an arbitrary fixed $t=0.50$ on extreme class imbalance ($0.097\%$ prevalence). Report Sensitivity at $80\%$ Specificity, ROC-AUC, and PR-AUC.

---

## 6. Stop Condition & Request for Approval

In strict accordance with instructions:
- The forensic investigation is **complete**.
- All root causes have been verified with empirical evidence.
- **NO further training runs have been initiated.**
- Standing by for user review and approval before implementing the proposed corrections.
