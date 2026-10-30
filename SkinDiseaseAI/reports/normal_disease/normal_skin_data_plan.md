# 📋 Normal-Skin Data Sourcing Plan & Scientific Evaluation Strategy
**Project:** SkinDiseaseAI — Normal Skin vs. Disease Skin Classification  
**Role:** Senior Machine Learning Engineer (Medical Computer Vision & Dataset Quality)  
**Date:** October 29, 2026  
**Status:** Stage A Review Complete — Normal Skin Sourcing Plan (Inspection & Planning Only — No Training)

---

## 1. Executive Summary & Problem Formulation

In Stage A, we established a foundational medical computer vision fact:  
**None of the standard dermatology datasets (Stanford DDI, Fitzpatrick17k, SCIN) contains verified, physician-annotated Normal / Healthy Skin controls.**

- **Stanford DDI (656 images):** 100% lesions (171 malignant, 485 benign).
- **Fitzpatrick17k (16,577 records):** 100% clinical conditions (`non-neoplastic`, `malignant`, `benign`).
- **SCIN (10,000+ images):** 97.9% active complaints; user `LOOKS_HEALTHY` is an unverified layperson self-report, and dermatologist `gradable == False` indicates ungradable photos, not healthy skin.

To build a clinically meaningful, generalizable **Class 0 (Normal Skin) vs. Class 1 (Possible Disease)** classifier for smartphone images, we must source genuine, verified normal skin without falling into the trap of **shortcut learning** (where an AI classifies lighting, background, or camera sensor differences instead of biological skin pathology).

---

## 2. In-Depth Investigation of Candidate Options

### Option A: Consented, Documented Primary Collection (Recommended Primary Strategy)
* **What it is:** A prospectively collected, protocol-governed dataset of clinical smartphone photographs of unaffected, normal human skin across diverse skin tones and body locations from consented volunteer participants.
* **Image Domain:** Native smartphone photography (Samsung, iPhone, OnePlus, Xiaomi) matching the exact inference environment of the mobile web app.
* **Target Body Locations:** 
  - Limbs: Forearms (dorsal/ventral), upper arms, back of hands, palms, shins, thighs.
  - Trunk: Upper chest, upper back (non-identifiable macro crops).
  - Neck and jawline (strictly excluding identifiable facial features).
* **Skin-Tone Diversity:** Stratified across the Monk Skin Tone (MST 1–10) and Fitzpatrick (FST I–VI) scales. In the context of Tamil Nadu and South Asia, this provides critical representation of FST IV, V, and VI, which are severely underrepresented in international dermatology benchmarks.
* **Strengths:** 
  - 100% ground-truth certainty: every image is known to come from an unaffected site.
  - Zero licensing restrictions: full ownership under research-study informed consent.
  - Known subject identifiers: guarantees 100% leak-proof, subject-independent splits.
* **Limitations:** Requires disciplined execution of collection, quality control, and de-identification.

---

### Option C: Public Healthy-Volunteer Control Datasets
We audited existing public datasets that claim to include "Normal Skin" or "Healthy Controls":

#### 1. Mendeley Data — *Skin Disease Classification Dataset* (Sharif et al.) / *32 Curated Categories (NORM class)*
- **Source:** Open research repository on Mendeley Data (`data.mendeley.com`), DOI: 10.17632.
- **Label Definition:** Folders titled `Healthy` or `Normal (NORM)` providing baseline skin patches.
- **Image Domain:** Mixed clinical photography and web-aggregated atlas images.
- **Access & License:** Open access under Creative Commons Attribution 4.0 (CC BY 4.0).
- **Critical Limitations:** 
  - **Unknown clinical provenance:** Images are web-scraped from online educational portals and blogs without biopsy, patient charts, or documented clinical protocols.
  - **Zero subject identifiers:** Cannot perform patient-level grouping; high risk of data leakage between training and testing.
  - **Demographic skew:** Almost exclusively Fitzpatrick I–II (fair skin tones).
  - **Artifact contamination:** Several samples contain watermarks, text overlays, or varying digital compressions.

#### 2. PAD-UFES-20 (Dermatological and Surgical Assistance Program, Brazil)
- **Source:** Mendeley Data, Pacheco et al. (2020), DOI: 10.17632/zr7vgbcyr2.1.
- **Image Domain:** 2,298 clinical smartphone images collected from 1,373 patients.
- **Access & License:** Open access (CC BY 4.0).
- **Critical Limitation for Class 0:** PAD-UFES-20 contains 6 lesion classes: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratosis (ACK), Seborrheic Keratosis (SEK), Melanocytic Nevus (NEV), and Melanoma (MEL). **It contains zero unaffected normal skin control images.** (Benign nevi are lesions, not normal skin).

#### 3. Verdict on Option C
Existing public "normal skin" folders on web repositories cannot serve as the primary ground truth because their provenance is unverified, subject metadata is absent, and skin tones are severely skewed toward fair skin. They may be retained only for secondary cross-dataset stress-testing, not for core model training.

---

### Option B: Lesion-Adjacent Background Skin (Dermoscopic Domain)
* **What it is:** Cropping the peripheral skin surrounding benign lesions in HAM10000 / ISIC dermoscopy images.
* **Why it CANNOT be used for the clinical mobile model:**
  1. **Optical Domain Mismatch:** Dermoscopy uses polarized contact immersion optics with fluid gels at 10× magnification. These look entirely different from non-contact smartphone cameras.
  2. **Field Cancerization & Subclinical Pathology:** In sun-exposed areas surrounding actinic keratoses or carcinomas, perilesional skin frequently harbors subclinical cellular atypia, solar elastosis, or inflammation.
  3. **Optical Artifacts:** Border zones often carry dermatoscope vignette rings, hair shadows, or gel bubble distortions.
* **Verdict:** Strictly quarantined as a separate, future dermoscopic research inquiry. It is excluded from the smartphone normal-vs-disease task.

---

## 3. Protocol for Consented Primary Collection (Option A)

To establish an unassailable, ethically sound, and medically rigorous dataset for **Class 0 (Normal Skin)**, the following 6-pillar protocol is defined:

```
[Participant Volunteer] 
       │
       ▼ (1. Digital / Paper Informed Consent)
[Informed Consent & Subject ID Assigned: SUBJ_XXX]
       │
       ▼ (2. Baseline Demographics: Age Group, Biological Sex, Monk Skin Tone 1-10, Fitzpatrick I-VI)
[Clinical Inclusion & Exclusion Screening]
       │
       ├─► Exclusion: Active rash, lesion, dermatitis, burn, bruise, scar, tattoo ──► [REJECT / ABSTAIN]
       │
       ▼ (3. Multi-Angle & Multi-Lighting Smartphone Photography)
[Controlled Image Acquisition (3 Lighting Setup + 2 Distances)]
       │
       ▼ (4. Automated Privacy & Quality Screening)
[Quality Assurance Gate: Laplacian Blur > 40, Exposure Check, Zero Facial Landmark Detection]
       │
       ▼ (5. Physician / Senior Reviewer Sign-off)
[Verified Class 0: Normal Skin Instance]
```

### Pillar 1: Informed Consent & Ethical Guidelines
- **Educational Research Scope:** Explicitly states the images will be used solely for non-commercial educational machine learning research and hackathon benchmarking.
- **Voluntary & Revocable:** Participation is voluntary with right to withdraw prior to dataset publication.
- **No Identifiable PII:** No names, phone numbers, or dates of birth are stored alongside image data.

### Pillar 2: Participant Privacy & De-Identification
- **Zero Facial / Biometric Features:** Strictly prohibits capturing eyes, lips, full face, ears, tattoos, distinctive scars, jewelry, or personal clothing.
- **Macro Patch Standard:** Images must be framed as localized skin patches (minimum resolution 1080×1080 px cropped to central 512×512 skin regions).
- **Pseudonymization:** Participants are assigned a randomized subject identifier (`SUBJ_001`, `SUBJ_002`, etc.).

### Pillar 3: Clinical Inclusion and Exclusion Criteria
- **Inclusion Criteria:**
  - Intact, unbroken skin surface.
  - Absence of visible erythema (redness), scaling, ulceration, edema, or active inflammation.
  - Absence of suspicious pigmented lesions, raised nodules, or palpable cysts in the frame.
  - Natural skin variations (fine vellus hair, standard skin pores, subtle non-dysplastic ephelides/freckles) are accepted as natural physiological characteristics.
- **Exclusion Criteria:**
  - Any active inflammatory dermatosis (eczema, psoriasis, acne vulgaris, urticaria).
  - Any fungal, bacterial, or viral infection (tinea, folliculitis, herpes simplex, warts).
  - Open abrasions, fresh scars, tattoos, or visible bruises.
  - Skin areas with active topical medications, heavy cosmetic foundations, or artificial self-tanning lotions.

### Pillar 4: Lighting & Optical Variations (To Prevent Confounding)
To ensure the AI learns skin biological texture rather than lighting conditions, each subject site must be photographed under **three distinct lighting environments**:
1. **Natural Diffused Daylight:** Near an indirect window during daytime (no harsh direct sun glare).
2. **Warm Indoor Incandescent / LED (approx. 3000K):** Standard residential warm lighting.
3. **Cool Indoor Fluorescent / White LED (approx. 5000K–6500K):** Standard commercial or clinical overhead lighting.
- **Framing Distances:**
  - Macro close-up: 10–15 cm from skin (sharp focal plane).
  - Medium clinical framing: 25–35 cm from skin (wider context).

### Pillar 5: Skin-Tone Diversity Target (Demographic Balance)
To prevent algorithmic bias against darker skin phenotypes, collection targets must enforce demographic quotas across the **Monk Skin Tone (MST 1–10)** scale:
- **Light Tones (MST 1–3 / FST I–II):** 25% of target samples.
- **Medium Tones (MST 4–6 / FST III–IV):** 35% of target samples.
- **Deep Tones (MST 7–10 / FST V–VI):** 40% of target samples (reflecting South Asian and global dark-skin demographics).

### Pillar 6: Two-Tier Review and Label Verification
1. **Automated Quality Filter:** Script checks focus (Laplacian variance $> 40$), luminance ($40 < L < 230$), and absence of blur.
2. **Reviewer Verification:** A secondary visual audit confirms that no subtle lesion, rash, or identifying artifact is present before setting `label_status = "verified_normal"`.

---

## 4. Leakage Prevention & Dataset Partitioning Plan

To guarantee academic rigor and prevent data leakage:

### A. Subject-Independent (Patient-Level) Grouping
- **The Golden Rule:** All images belonging to `SUBJ_001` (across all body sites and all lighting setups) must be assigned **exclusively to ONE partition** (Train, Validation, OR Test).
- Under no circumstances will different photos of the same subject appear in both Train and Test splits.

### B. Split Ratios
- **Train Split (70% of subjects):** Used for representation learning, feature extraction, and class weighting.
- **Validation Split (15% of subjects):** Used strictly for model checkpoint selection, threshold calibration, and hyperparameter tuning.
- **Frozen Test Split (15% of subjects):** Preserved untouched. Evaluated **ONCE** at the end of the project.

### C. Stratified Group Splitting
- Split selection will use `GroupShuffleSplit` (grouped by `subject_id`) stratified across `skin_tone` categories to ensure all three splits (Train, Val, Test) possess equal proportions of dark, medium, and light skin tones.

### D. Multi-Modal Deduplication Auditing
- **Exact Hash Matching:** SHA-256 hashes generated for all candidate images to ensure zero identical files exist across splits.
- **Perceptual Hash Matching (pHash):** Hamming distance threshold $\le 4$ used to flag and exclude accidental near-duplicate photos of the same skin area.

---

## 5. Proposed Dual-Source Architecture for Normal vs. Disease

```
┌────────────────────────────────────────────────────────┐
│            PROPOSED BINARY TRAINING CORPUS             │
├──────────────────────────┬─────────────────────────────┤
│   CLASS 0: NORMAL SKIN   │   CLASS 1: POSSIBLE DISEASE │
├──────────────────────────┼─────────────────────────────┤
│ Source: Option A         │ Source: Stanford DDI + SCIN │
│ Protocol-Governed        │ Validated Clinical Disease  │
│ Consented Volunteers     │ Photos (Allergies, Tumors,  │
│ Multi-tone (FST I - VI)  │ Infections, Eczema, Nevi)   │
│ Smartphone Macro Images  │ Smartphone & Clinical Cams  │
└──────────────────────────┴─────────────────────────────┘
                           ▲
                           │
             ┌─────────────┴─────────────┐
             │ UNCERTAIN / UNSUITABLE    │
             ├───────────────────────────┤
             │ Abstention Gate:          │
             │ - Blurry / low contrast   │
             │ - Non-skin objects        │
             │ - Conflicting multi-lesion│
             └───────────────────────────┘
```

- **Class 0 (Normal Skin):** Consented volunteer smartphone images adhering to the 6-pillar protocol.
- **Class 1 (Possible Skin Disease):** Confirmed pathological lesions and conditions from Stanford DDI (clinical macro) and confirmed clinical cases from Google SCIN / Fitzpatrick17k.
- **Abstention Outcome (`UNCERTAIN_OR_UNSUITABLE`):** Inputs failing quality checks, non-skin images, or ungradable presentations are rejected before reaching binary classification.

---

## 6. Next Steps & Required User Approvals

Stage A review is complete. All findings and planning frameworks are recorded in this document.

### Actions Deferred until Explicit Approval:
- [ ] No image downloads have been initiated.
- [ ] No labels have been merged or created.
- [ ] No train/val/test splits have been generated.
- [ ] No machine learning models have been trained.

Please review this plan and let me know:
1. Do you approve the **6-pillar collection protocol for Option A**?
2. What target sample size for normal skin volunteer contributions would you like to plan for (e.g., 50–100 distinct subjects with 3–5 images each = 200–400 normal patches)?
3. Once approved, shall we proceed to **Stage B (Formal Label Mapping & Split Plan)**?
