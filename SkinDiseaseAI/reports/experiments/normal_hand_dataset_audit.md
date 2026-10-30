# Normal-Looking Hand Skin Dataset Audit (Stage A)

**Date:** October 29, 2026  
**Auditor:** Senior Deep Learning Engineer & Medical Computer Vision Specialist  
**Audited Location:** `C:\Users\USER\Downloads\normal skin dataset`  
**Target Application:** SkinDiseaseAI — Normal-Looking Hand Skin Model  
**Current Milestone:** Stage A — Dataset Audit & Inspection (Pre-Training Gate)  

---

## 1. Executive Summary & Dataset Topology

An exhaustive programmatic and visual inspection was conducted recursively across `C:\Users\USER\Downloads\normal skin dataset`. 

The directory does **not** contain a single monolithic hand dataset. Instead, it contains **three distinct unpacked archives** spanning 16,423 total files (16,420 images, 2 CSV metadata files, and 1 text file):

| Archive Subfolder | Dataset Origin / Identification | Total Images | Modality / Body Region | Suitability for Hand Model |
| :--- | :--- | :---: | :--- | :--- |
| **`archive (3)\Hands\Hands`** | **11k Hands Dataset** (Afifi, 2019) | **11,076** | **Dorsal & Palmar Hands** on white studio background | **Primary Target Hand Dataset** |
| `archive (3)` (Numbered `1`–`15`) | Facial Skin Condition Dataset | 45 | Close-up facial portraits (15 subjects) | Unsuitable (Non-hand) |
| `archive (2)\Oily-Dry-Skin-Types` | Facial Cosmetic Skin Types | 3,152 | Facial foreheads, cheeks, noses (`dry`, `normal`, `oily`) | Unsuitable (Non-hand) |
| `archive (4)` | Eczema vs Tinea Disease Collection | 2,147 | Multi-site clinical lesions (arms, legs, feet, torso, hands) | **Severe confounding risk** (only ~8% hands) |

**Key Finding:** The intended dataset for hand skin modeling is the **11k Hands** collection located at `archive (3)\Hands\Hands` with ground-truth metadata in `archive (3)\HandInfo.csv`.

---

## 2. Quantitative Verification of 11k Hands (`archive (3)`)

Every single image file and metadata row was programmatically validated:

* **Total Images on Disk:** **`11,076`**
* **Total Metadata Records in [HandInfo.csv](file:///C:/Users/USER/Downloads/normal%20skin%20dataset/archive%20(3)/HandInfo.csv):** **`11,076`**
* **Alignment:** Exactly **`100.0%`** match between disk filenames and CSV entries (0 missing on disk, 0 unindexed in CSV).
* **Image Format:** 100% standard JPEG (`.jpg`), 3-channel RGB.
* **Image Resolution:** 100% strictly uniform **`1600 × 1200`** pixels (4:3 aspect ratio).
* **Corrupt / Unreadable / 0-byte Files:** **`0`** (All 11,076 images successfully decoded).
* **Blank / Low-Variance Files:** **`0`**.

### Exact Bit-for-Bit Duplicate Scan
An MD5 hash computed across all 11,076 files revealed **11,075 unique hashes** and exactly **one byte-for-byte duplicate pair**:
* **MD5:** `1f2c97dee29d90dff358c1d0ffd6d6d8`
* **File 1:** `Hand_0009558.jpg`
* **File 2:** `Hand_0009559.jpg`
* **Metadata:** Both belong to Subject `4000000` (19-year-old female, dorsal right hand, no accessories).
* **Recommendation:** Exclude `Hand_0009559.jpg` from the experimental manifest in Stage B.

---

## 3. Demographics, Anatomical Aspects, and Leakage Risk

### A. Critical Finding: Participant Clumping & Leakage Danger
* **Total Unique Participants (`id`):** **`189` distinct individuals**.
* **Images per Participant:** Mean **`58.6 ± 30.8`** images per person (Median: 52, Min: 14, Max: 187).
* **Severe Data Leakage Alert:** If this dataset were split randomly at the image level, near-identical photos of the exact same hand (taken seconds apart) would be distributed across training, validation, and test splits, artificially inflating validation accuracy.
* **Mandatory Rule for Stage B:** **Splits MUST be strictly partitioned by Participant ID (`id`)**, never by image.

### B. Anatomical Aspects (Dorsal vs Palmar)
The dataset is well-balanced across hand surfaces and left/right symmetry:
* **Dorsal Right** (Back of right hand): **`2,892`** (26.11%)
* **Dorsal Left** (Back of left hand): **`2,788`** (25.17%)
* **Palmar Right** (Palm of right hand): **`2,813`** (25.40%)
* **Palmar Left** (Palm of left hand): **`2,583`** (23.32%)
* **Total Dorsal (Back):** 5,680 (51.28%)
* **Total Palmar (Palm):** 5,396 (48.72%)

### C. Demographic Distributions
* **Gender:**
  * Female: **`7,109`** (64.18%)
  * Male: **`3,967`** (35.82%)
* **Age Profile:**
  * Mean: **22.5 years** (std: 6.9, range: 18 to 75 years).
  * 25th percentile: 21 years | Median: 21 years | 75th percentile: 22 years.
  * **Demographic Bias:** Heavy concentration of young adults / university students; hands with pronounced wrinkles, age spots, or solar elastosis are severely underrepresented.
* **Skin Tone Annotations:**
  * `medium`: **`6,495`** (58.64%)
  * `fair`: **`3,493`** (31.54%)
  * `dark`: **`758`** (6.84%)
  * `very fair`: **`330`** (2.98%)

---

## 4. Confounders, Visual Artefacts, and Shortcut Risks

Visual and automated feature analysis identified three critical confounding factors that must be addressed during preprocessing and modeling:

### 1. Studio Background Shortcut (Uniform White Board)
* Every hand in 11k Hands was captured against a flat, white matte background board.
* Measured Corner Pixel Values (RGB): **`R = 253.6 ± 1.2, G = 253.4 ± 1.1, B = 253.2 ± 1.2`**.
* **Risk:** A convolutional neural network trained on raw whole images will learn to detect *"image with white border"* rather than learning hand skin features.
* **Mitigation:** Stage D/E models must utilize **hand bounding-box crops** or **hand skin segmentation masks** rather than uncropped 1600×1200 frames.

### 2. Accessories & Jewellery (`accessories == 1`)
* **`3,211` images (28.99%)** contain rings, wristwatches, or bracelets.
* **`7,865` images (71.01%)** have zero accessories.
* **Risk:** The network could correlate jewellery reflections with specific participants or mistake metallic rings for lesions/anomalies.

### 3. Nail Polish (`nailPolish == 1`)
* **`1,067` images (9.63%)** contain painted fingernails (red, black, patterned polish).
* **`10,009` images (90.37%)** contain unpainted nails.
* **Risk:** Artificial pigmentation on nail beds can distort skin chromaticity algorithms and confuse texture modeling.

---

## 5. Clinical Ground Truth & "Irregularities" Audit

### A. Important Medical Reality
**The 11k Hands dataset was created for biometric identity and gender recognition, NOT as a dermatological reference standard.**
* Images were not collected by dermatologists and have no biopsy or clinical examination records.
* Therefore, the "0" label in `irregularities` means **"visually typical hand without prominent gross deformity"**, NOT "medically certified healthy skin".
* It is scientifically dishonest to market a model trained on this dataset as a "skin disease ruling-out tool".

### B. Audit of Irregularities (`irregularities == 1`)
In `HandInfo.csv`, exactly **`164` images (1.48%)** are flagged with `irregularities = 1`.

Programmatic grouping reveals that all 164 images belong to just **4 specific participants**:

| Participant ID | Gender | Age | Skin Color | Total Images with Irregularity | Hand Aspects Affected | Notes / Observed Features |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`27`** | Male | 23 | Fair | 21 | Dorsal right (11), Palmar right (10) | Visible scar / surface irregularity |
| **`570`** | Female | 21 | Medium | 21 | Dorsal right (10), Palmar right (5), Palmar left (3), Dorsal left (3) | Discrete skin lesion / birthmark |
| **`1566`** | Male | 21 | Medium | 99 | Palmar left (35), Dorsal right (24), Dorsal left (22), Palmar right (18) | Prominent epidermal marks / scarring |
| **`1000000`** | Male | 70 | Dark | 23 | Dorsal right (23) | Age-related skin texture, prominent veins, spots |

### C. Manual Review List for the User
A dedicated inspection list has been compiled for these 4 subjects. They must **not** be included in the normal baseline training split:
1. `Hand_0000774.jpg` to `Hand_0000794.jpg` (Subject 27)
2. `Hand_0001740.jpg` to `Hand_0001760.jpg` (Subject 570)
3. `Hand_0004120.jpg` to `Hand_0004218.jpg` (Subject 1566)
4. `Hand_0010970.jpg` to `Hand_0010992.jpg` (Subject 1000000)

---

## 6. Audit of Accompanying Folders (`archive (2)` & `archive (4)`)

### A. `archive (2)\Oily-Dry-Skin-Types` (3,152 Images)
* **Contents:** Facial portraits classified into `dry` (758), `normal` (1,274), and `oily` (1,120).
* **Verdict:** Unrelated to hand skin. Should be excluded from the hand model pipeline.

### B. `archive (4)` (Eczema vs Tinea, 2,147 Images)
* **Contents:** Clinical photographs of eczema (1,122) and tinea (1,025).
* **MediaPipe Hand Detection Audit (Sample N=60):**
  * Eczema sample: Only **`13.3%`** (4/30) contained hands.
  * Tinea sample: Only **`3.3%`** (1/30) contained hands.
* **Severe Trap Warning:** The vast majority of images in `archive (4)` show arms, feet, legs, scalps, or torsos in clinical hospital lighting.
* **Scientific Verdict:** Pairing `archive (4)` against `archive (3)` to build a "Normal vs Disease" binary classifier would result in a **fatal confounding shortcut**: the classifier would learn to distinguish "hands on white paper" vs "torso/legs on hospital sheets", rather than normal skin vs disease. `archive (4)` cannot be used as a hand disease benchmark without individual manual curation of verified hand lesions.

---

## 7. Recommendations for Subsequent Stages

### Recommendation 1 (Stage B: Manifest Construction)
1. Use `archive (3)\Hands\Hands` as the sole source for the normal hand skin model.
2. Build `data/experiments/normal_hand_v1/manifest_v1.csv` with columns: `image_path`, `participant_id`, `aspect`, `skin_color`, `accessories`, `nail_polish`, `irregularities`, and `split`.
3. Remove exact duplicate `Hand_0009559.jpg`.
4. Partition the 189 participants into **Subject-Disjoint Splits** (e.g., 132 Train, 28 Validation, 29 Test).
5. Quarantine the 4 irregular participants (`27`, `570`, `1566`, `1000000`) into a separate evaluation set to test whether the normal model flags them as anomalous.

### Recommendation 2 (Stage C: Task Formulation)
* **Adopt Option C / Option D (Suitability Gate & Normal-Hand Density/Anomaly Modeling):**
  * Train an embedding-based feature extractor (MobileNetV3 or EfficientNetB0) on clean normal hand crops.
  * Learn the distribution of normal hand skin texture and appearance.
  * Treat high Mahalanobis distance / low likelihood as an indicator that the image *differs from the normal training distribution* (which could be a lesion, rash, artifact, or poor lighting).
  * **Strict Disclaimer:** Explicitly display: *"Image differs from standard normal hand skin dataset baseline. This is an experimental visual check, NOT a medical diagnosis."*

---

## 8. Checkpoint Status

**Stage A is complete.**
* No production files, models, or configurations were modified.
* All data was audited programmatically from the local filesystem.
* **Next Step:** Awaiting user review and formal approval of this audit before beginning Stage B (Manifest Generation).
