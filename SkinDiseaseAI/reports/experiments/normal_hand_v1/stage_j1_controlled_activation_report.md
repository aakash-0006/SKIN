# SkinDiseaseAI — Stage J.1: Controlled Activation Acceptance Report

**Date:** 2026-10-30  
**Status:** Completed & Validated — Local Acceptance Testing Successful  
**Live Feature Flag Status:** `ENABLE_NORMAL_HAND_DETECTOR = False` (Preserved / Verified)  
**Safety Classification:** Educational research prototype — NOT a medical device  

---

## 1. Executive Summary

This report documents the local, controlled, and fully reversible acceptance testing of the integrated **Candidate 1B normal-hand appearance detector** (`models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz`) as part of **Stage J.1**.

### Primary Verification Findings
1. **Real Prediction Path Verification:** All 8 mandated test scenarios (typical hand, hand with accessories, hand with nail polish, complex lighting/background, non-hand gaming poster, malformed bytes, supported dermoscopic close-up, and missing candidate artifact fallback) were executed through the integrated `src.predict.predict_bytes` interface.
2. **Strict Disease Suppression on Hands:** When macro-camera hands are evaluated through Candidate 1B, all 7-class dermoscopic disease classification outputs (`scores = []`, `code = None`, `name = None`, `melanoma_flag = False`) are strictly suppressed.
3. **Preservation of Dermoscopy Close-up Path:** Dermoscopic close-ups (e.g. HAM10000 test partition sample `ISIC_0032417`) do not trigger the hand pipeline and route exclusively to the 7-class lesion classifier (`best_model.keras` + `ood_gate.npz`), preserving existing clinical classification and OOD gating.
4. **Resilient Fallback on Missing Artifact:** When the Candidate 1B artifact is unavailable, the pipeline deterministically falls back to the baseline detector (`hand_anomaly_baseline.npz`, $k=48$, unnormalized, threshold 8.768) without error or false disease prediction.
5. **Non-Hand & Corrupt Rejection:** Non-hand artwork (gaming poster) and corrupt byte inputs are rejected safely by the domain gate and validation layer without producing disease labels or crashes.
6. **Integrity & Zero-Regression:** 
   - Dedicated Candidate 1B test suite: **9/9 tests passed** in 3.172s.
   - Full regression suite: **56/56 tests passed** in 10.062s.
   - All 7 protected repository assets, the baseline artifact, and the Candidate 1B artifact match their recorded SHA-256 checksums bit-for-bit.
   - `app.py`, camera hardware logic, and UI layouts were **not modified**.
   - Quarantined participants (27, 570, 1566, 1000000) were **never accessed**.
   - `ENABLE_NORMAL_HAND_DETECTOR = False` remains strictly enforced in the ordinary application.

---

## 2. Test Environment & Isolation Protocol

Testing was performed locally using the active Python virtual environment:
- **Python Binary:** `.\.venv\Scripts\python.exe`
- **Interpreter Version:** Python 3.11.9
- **Working Directory:** `d:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI`
- **Isolation Mechanism:** An isolated programmatic test harness (`scratch/run_stage_j1_acceptance.py`) temporarily set `predict_module.ENABLE_NORMAL_HAND_DETECTOR = True` exclusively in-memory within a `try...finally` block. Upon completion of each case and script termination, the live module flag was verified restored to `False`.
- **Git State:** Local repository root is not a Git repository. Source state preservation was confirmed via byte-level backups in `backups/stage_j_integration_backup/` and cryptographic SHA-256 verification against the locked master backup `D:\DOWNLOADS D\SkinDiseaseAI_v1_20261027.zip`.

---

## 3. Real Prediction Path Verification (8 Acceptance Cases)

All test cases were executed by calling `src.predict.predict_bytes(data, require_hand_validation=True, validation_mode="all_skin")`.

### Summary Results Table

| Case # | Description | Source / Input Identifier | Function Invoked | Routing Path | Artifact Loaded | L2 Norm | Anomaly Score / Threshold | Decision / Verdict | Disease Scores Suppressed | Displayed User Message |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | Typical Hand Photo (Clean) | 11k Hands test partition: `Hand_0000002.jpg` (Participant 0) | `src.predict.predict_bytes` | Macro Hand Pipeline | Candidate 1B ($k=32$) | True | 4.787 / 6.931 | `USABLE_HAND_IMAGE` | Yes (`code=None`, `scores=[]`) | *"Image appearance is consistent with reference normal hand skin dataset (score: 4.79 <= threshold 6.93). Notice: A typical visual appearance does not guarantee medical health."* |
| **2** | Hand with Accessories / Jewellery | 11k Hands test partition: `Hand_0000021.jpg` (Participant 0, accessories=1) | `src.predict.predict_bytes` | Macro Hand Pipeline | Candidate 1B ($k=32$) | True | 5.183 / 6.931 | `USABLE_HAND_IMAGE` | Yes (`code=None`, `scores=[]`) | *"Image appearance is consistent with reference normal hand skin dataset (score: 5.18 <= threshold 6.93). Notice: A typical visual appearance does not guarantee medical health."* |
| **3** | Hand with Nail Polish | 11k Hands test partition: `Hand_0001096.jpg` (Participant 1535, nail_polish=1) | `src.predict.predict_bytes` | Macro Hand Pipeline | Candidate 1B ($k=32$) | True | 5.221 / 6.931 | `USABLE_HAND_IMAGE` | Yes (`code=None`, `scores=[]`) | *"Image appearance is consistent with reference normal hand skin dataset (score: 5.22 <= threshold 6.93). Notice: A typical visual appearance does not guarantee medical health."* |
| **4** | Hand in Unusual Lighting / Complex Background | Fixture: `webcam_classroom_hand.jpg` | `src.predict.predict_bytes` | Macro Hand Pipeline | Candidate 1B ($k=32$) | True | 10.167 / 6.931 | `UNUSUAL_RELATIVE_TO_TRAINING_DATA` | Yes (`code=None`, `scores=[]`) | *"Appearance differs from reference normal hand dataset (score: 10.17 > threshold 6.93). This visual variation may be caused by a lesion, rash, unusual lighting, angle, or foreign object. Notice: This is an experimental visual check, NOT a medical diagnosis."* |
| **5** | Non-Hand Image (Gaming Poster) | Fixture: `gaming_poster.jpg` | `src.predict.predict_bytes` | Stage 1 Domain Gate Rejection | None (rejected before model) | N/A | None / None | `None` (`stage_failed: domain_gate`) | Yes (`code=None`, `scores=[]`) | *"No condition label shown. This photo is outside the validated domain (dermoscopy)."* |
| **6** | Corrupt / Unreadable Binary Stream | In-memory synthetic malformed stream: `b'NOT_A_VALID_IMAGE_HEADER_12345\x00\xff\xfe'` | `src.predict.predict_bytes` | Stage 1 Domain Gate Rejection | None (rejected before model) | N/A | None / None | `None` (`stage_failed: domain_gate`) | Yes (`code=None`, `scores=[]`) | *"No condition label shown. This photo is outside the validated domain (dermoscopy)."* |
| **7** | Supported Dermoscopic Close-up | HAM10000 test partition: `ISIC_0032417.jpg` (ground truth: bkl) | `src.predict.predict_bytes` | Dermoscopy 7-Class Disease Pipeline | `best_model.keras` + `ood_gate.npz` | False (N/A) | None / None | `None` (Hand detector not applied) | No (7 class scores computed: `code=bkl`) | Disease predictions rendered with dermoscopic caveats; hand detector completely bypassed. |
| **8** | Missing Candidate 1B Artifact (Fallback) | Simulated missing candidate artifact; Input: `Hand_0000002.jpg` | `src.predict.predict_bytes` | Macro Hand Pipeline | Baseline fallback (`hand_anomaly_baseline.npz`, $k=48$) | False | 5.520 / 8.768 | `USABLE_HAND_IMAGE` | Yes (`code=None`, `scores=[]`) | *"Image appearance is consistent with reference normal hand skin dataset (score: 5.52 <= threshold 8.77). Notice: A typical visual appearance does not guarantee medical health."* |

---

## 4. Case-by-Case Technical Observations

### Case 1: Typical Hand Photograph (Clean, No Accessories)
- **Input:** Participant 0, `Hand_0000002.jpg` (test split).
- **Behavior:** MediaPipe landmark detection succeeded; cropped to $224 \times 224$; MobileNetV2 feature vector was $L_2$-normalized; projected via 32-component PCA; Mahalanobis distance calculated as **4.787** against threshold **6.931**.
- **Clinical Safety:** 7-class lesion scores completely suppressed (`scores: []`, `code: None`, `melanoma_flag: False`). User message explicitly warns: *"Notice: A typical visual appearance does not guarantee medical health."*

### Case 2: Hand with Jewellery or Accessories
- **Input:** Participant 0, `Hand_0000021.jpg` (accessories = 1, test split).
- **Behavior:** Candidate 1B produced an anomaly score of **5.183**, remaining safely under the 6.931 threshold.
- **Improvement over Baseline:** Under the baseline unnormalized model, accessory images exhibited elevated disparity. Candidate 1B successfully avoided false anomaly escalation while maintaining strict suppression of disease classifications.

### Case 3: Hand with Nail Polish
- **Input:** Participant 1535, `Hand_0001096.jpg` (nail_polish = 1, test split).
- **Behavior:** Landmark detection located the dorsal hand region. Extracted $L_2$-normalized embedding yielded an anomaly score of **5.221** (threshold: 6.931), categorizing the image as `USABLE_HAND_IMAGE`.
- **Finding:** Polish presence did not artificially spike the Mahalanobis distance beyond the normal threshold under $L_2$ normalization.

### Case 4: Hand in Unusual Lighting / Complex Background
- **Input:** `tests/fixtures/webcam_classroom_hand.jpg`.
- **Behavior:** Hand was detected by MediaPipe, but non-uniform ambient classroom lighting and cluttered background produced an anomaly score of **10.167** (> 6.931).
- **Decision:** Correctly categorized as `UNUSUAL_RELATIVE_TO_TRAINING_DATA`.
- **Clinical Safety:** Did NOT output any disease label or melanoma warning. Replaced with educational advisory: *"This visual variation may be caused by a lesion, rash, unusual lighting, angle, or foreign object. Notice: This is an experimental visual check, NOT a medical diagnosis."*

### Case 5: Non-Hand Image (Gaming Poster)
- **Input:** `tests/fixtures/gaming_poster.jpg`.
- **Behavior:** MediaPipe hand detection failed (`crop_res.hand_count == 0`). Input fell through to Stage 1 Domain Gate, which rejected it with reason: *"the image does not look like the dermoscopic lesion images this model was trained on"*.
- **Result:** Safely rejected at Stage 1. Disease classifier was never invoked.

### Case 6: Corrupt or Unreadable Input
- **Input:** Binary stream with invalid header bytes (`b'NOT_A_VALID_IMAGE_HEADER_12345\x00\xff\xfe'`).
- **Behavior:** `cv2.imdecode` returned `None`. Hand detection was bypassed. Domain gate rejected the byte stream safely with reason: *"The file is not a readable image."*
- **Result:** No unhandled exceptions or crashes.

### Case 7: Supported Dermoscopic Close-up (HAM10000 Test Split)
- **Input:** `D:\DOWNLOADS D\archive\HAM10000_images_part_2\ISIC_0032417.jpg` (HAM10000 test split; diagnosis `bkl`).
- **Behavior:** MediaPipe hand detector identified 0 hands (`crop_res.hand_count == 0`). Image proceeded to Stage 1 Domain Gate (passed) and anti-spoof/skin checks (passed).
- **Routing:** Reached 7-class lesion classifier (`best_model.keras`). Predicted class: `bkl` (Benign keratosis-like lesions) with full probability distribution (7 scores). Hand anomaly score was **not applied**.
- **Integrity:** Dermoscopic inference pipeline is completely insulated from macro-camera hand logic.

### Case 8: Missing or Corrupted Candidate 1B Artifact
- **Simulation:** In-memory mock returning `False` for `candidate_1b_l2_mahalanobis_k32.npz` existence in `src.predict.get_normal_hand_pipeline`.
- **Behavior:** `get_normal_hand_pipeline` detected absence of Candidate 1B and seamlessly fell back to `hand_anomaly_baseline.npz`.
- **Configuration Loaded:** Components: $k=48$, `normalize_features: False`, threshold: `8.768`.
- **Inference on Hand_0000002.jpg:** Yielded score **5.520** <= 8.768 (`USABLE_HAND_IMAGE`). 7-class disease predictions remained completely suppressed.

---

## 5. Regression & Integrity Verification

### Test Suite Execution Commands & Outputs

```powershell
# 1. Full Regression Suite
& .\.venv\Scripts\python.exe -m unittest discover -s tests -t .
# Result: Ran 56 tests in 10.062s. OK.

# 2. Dedicated Stage J Candidate 1B Test Module
& .\.venv\Scripts\python.exe -m unittest tests.test_stage_j_candidate_1b
# Result: Ran 9 tests in 3.172s. OK.
```

### Cryptographic Hash Verification Table

All files were verified by computing SHA-256 over raw file bytes:

| Asset Name | Relative Path | Target Role | Verified SHA-256 Checksum | Match Status |
|:---|:---|:---|:---|:---:|
| Best Model Keras | `models/best_model.keras` | 7-Class Disease Classifier | `C6AF0869563F708D1DC26E67668F0838D7EDFD277B7B585FA65527BCFFDDF2FD` | **MATCH** |
| ISIC Finetuned | `models/isic_finetuned.keras` | Alternative Model | `6149DBD49D5DC403CF81E55BA118B4CD20D70D4F7A0098617BAE994962005FF2` | **MATCH** |
| ISIC Augmented v1 | `models/isic_augmented_v1.keras` | Alternative Model | `FE3EB872F919BCDAA4AE69C318993B50A33C0FEE815D00AADD2BAB75ADA99D34` | **MATCH** |
| OOD Gate | `models/ood_gate.npz` | Dermoscopy Domain Gate | `DCD9974A82E40E4DECFB1170893B7934C9BAB9B05C9DF316046F25EA47479DBE` | **MATCH** |
| Splits Manifest | `outputs/splits.csv` | HAM10000 Splits | `8DEEDC0338273B81EAD6BC89D8C12D53E6091906390EC81C5EC9CE37170CB3EC` | **MATCH** |
| Combined Splits | `outputs/combined_splits.csv` | Multi-dataset Splits | `07F94433D6BB638EDE552EF9338F0E2258BEAFBEFB85F487CDC58A6DD7548149` | **MATCH** |
| Config JSON | `config.json` | Global Configuration | `AD23CC09A5050C5B390AF56BCE25B389CAC72E1BB52E3416F8A48C08801DEDB1` | **MATCH** |
| Baseline Anomaly Detector | `models/experiments/normal_hand_v1/hand_anomaly_baseline.npz` | Fallback Detector | `5B2B53451932ACC9644AB612AA0AF7FF0033E8A03ED4503C8D2EF686DCB8DBF4` | **MATCH** |
| Candidate 1B Detector | `models/experiments/normal_hand_v1/stage_g/candidate_1b_l2_mahalanobis_k32.npz` | Integrated Detector | `C4F0FED96CB0308C8155CED51607E0DF872F4C91EC2180ADCD4ED311267FC5AB` | **MATCH** |

### Quarantine Compliance Verification
- Manifest query strictly filtered against quarantined participants `{27, 570, 1566, 1000000}`.
- Test partition samples used:
  - Participant 0 (`Hand_0000002.jpg`, `Hand_0000021.jpg`)
  - Participant 1535 (`Hand_0001096.jpg`)
- None of the 243 quarantined images were loaded or accessed during testing.

### Ordinary Application State Verification
- `app.py` SHA-256: `315D1C948A6E5010AD15970F5713B1E7A587687AB5F55ECD080B2CC41BE1E773` (unchanged).
- Live feature flag in `src/predict.py`:
  ```python
  ENABLE_NORMAL_HAND_DETECTOR = False  # Verified False
  ```
- Ordinary application startup remains strictly in the baseline dermoscopy screening mode.

---

## 6. Identified Limitations & Test Coverage Notes

1. **MediaPipe Landmark Orientation Sensitivity:**
   - In 11k Hands test images where hands are placed at unusual angles or fingers are partially cut off by image borders (e.g., `Hand_0001091.jpg`), MediaPipe's hand landmark pipeline can fail detection. When detection fails, the input is treated as a general skin image rather than a localized macro-hand image.
2. **Lighting Variations:**
   - As demonstrated in Case 4, extreme lighting deviations (e.g. harsh fluorescent classroom lighting) trigger `UNUSUAL_RELATIVE_TO_TRAINING_DATA`. This is safe from a medical standpoint (prevents false reassurance), but users in poorly lit environments should receive guidance on standard photography conditions.
3. **Clinical Scope:**
   - The detector checks visual appearance against a reference normal dataset; it is an appearance-consistency classifier, NOT a medical clearance tool. All displayed messages explicitly reiterate this caution.

---

## 7. Rollback Instructions

If any rollback is required:
1. Restore `src/predict.py` from backup:
   ```powershell
   Copy-Item "backups/stage_j_integration_backup/predict.py.bak" "src/predict.py" -Force
   ```
2. Restore `src/normal_hand/pipeline.py` from backup:
   ```powershell
   Copy-Item "backups/stage_j_integration_backup/pipeline.py.bak" "src/normal_hand/pipeline.py" -Force
   ```
3. Remove test module:
   ```powershell
   Remove-Item "tests/test_stage_j_candidate_1b.py" -Force
   ```
4. Confirm test suite passes:
   ```powershell
   & .\.venv\Scripts\python.exe -m unittest discover -s tests -t .
   ```

---

## 8. Conclusion & Sign-Off Condition

Stage J.1 local acceptance testing has fully validated the integration, routing behavior, suppression guarantees, fallback robustness, and regression immunity of Candidate 1B.

**Current State:**
- `ENABLE_NORMAL_HAND_DETECTOR = False` in `src/predict.py`.
- No protected models or configurations modified.
- No live application deployment or external publication initiated.
- **STOPPED and awaiting user review and instructions.**
