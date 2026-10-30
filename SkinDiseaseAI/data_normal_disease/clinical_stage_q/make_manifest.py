"""Build Stage Q local clinical dataset manifest and patient-level splits."""
import os
import sys
import pandas as pd
import numpy as np

def main():
    print("=== Stage Q: Building Local Clinical Dataset Manifest ===")
    records = []

    # 1. Stanford DDI (656 images)
    print("1. Adding Stanford DDI...")
    ddi_dir = r"D:\DOWNLOADS D\ddidiversedermatologyimages"
    ddi_df = pd.read_csv(os.path.join(ddi_dir, "ddi_metadata.csv"))
    for _, row in ddi_df.iterrows():
        f_path = os.path.join(ddi_dir, row["DDI_file"])
        records.append({
            "source_dataset": "Stanford_DDI",
            "image_id": f"ddi_{row['DDI_ID']}",
            "image_path": f_path,
            "patient_id": f"ddi_pt_{row['DDI_ID']}",
            "coarse_label": "abnormal_condition",
            "fine_label": row["disease"],
            "is_malignant": bool(row["malignant"]),
            "skin_tone": str(row["skin_tone"]),
            "body_site": "unspecified",
            "dataset_role": "independent_clinical_eval"
        })

    # 2. Clinical Eczema vs Tinea (2,147 images)
    print("2. Adding Eczema vs Tinea...")
    ecz_base = r"C:\Users\USER\Downloads\normal skin dataset\archive (4)"
    for split_folder in ["train", "test"]:
        for cond in ["eczema", "tinea"]:
            cond_dir = os.path.join(ecz_base, split_folder, cond)
            if os.path.exists(cond_dir):
                for fname in os.listdir(cond_dir):
                    f_path = os.path.join(cond_dir, fname)
                    records.append({
                        "source_dataset": "Eczema_vs_Tinea",
                        "image_id": f"ecztin_{cond}_{fname.split('.')[0]}",
                        "image_path": f_path,
                        "patient_id": f"ecztin_{fname.split('.')[0]}",
                        "coarse_label": "abnormal_condition",
                        "fine_label": cond,
                        "is_malignant": False,
                        "skin_tone": "unrecorded",
                        "body_site": "unspecified",
                        "dataset_role": "clinical_condition_training"
                    })

    # 3. 11k Normal Hands (11,076 images)
    print("3. Adding 11k Normal Hands...")
    hand_base = r"C:\Users\USER\Downloads\normal skin dataset\archive (3)"
    hand_df = pd.read_csv(os.path.join(hand_base, "HandInfo.csv"))
    hands_img_dir = os.path.join(hand_base, "Hands", "Hands")
    for _, row in hand_df.iterrows():
        f_path = os.path.join(hands_img_dir, row["imageName"])
        is_normal = (row["irregularities"] == 0)
        records.append({
            "source_dataset": "11k_Normal_Hands",
            "image_id": f"hand_{row['id']}_{row['imageName'].split('.')[0]}",
            "image_path": f_path,
            "patient_id": f"hand_subj_{row['id']}",
            "coarse_label": "normal_looking_skin" if is_normal else "hand_irregularity",
            "fine_label": "healthy_skin" if is_normal else "irregularity",
            "is_malignant": False,
            "skin_tone": str(row["skinColor"]),
            "body_site": "hand",
            "dataset_role": "normal_hand_reference"
        })

    df = pd.DataFrame(records)
    print(f"\nTotal records: {len(df)}")
    print(df["source_dataset"].value_counts())
    print("\nCoarse labels:")
    print(df["coarse_label"].value_counts())

    # Patient-level splitting for normal hands and clinical conditions
    print("\n=== Generating Patient-Level Splits ===")
    rng = np.random.RandomState(42)
    df["split"] = "unassigned"

    # Stanford DDI: 100% held out for independent clinical evaluation
    df.loc[df["source_dataset"] == "Stanford_DDI", "split"] = "independent_test"

    # 11k Normal Hands: Grouped by subject id (190 subjects)
    hand_subjects = df[df["source_dataset"] == "11k_Normal_Hands"]["patient_id"].unique()
    rng.shuffle(hand_subjects)
    n_subj = len(hand_subjects)
    n_train = int(n_subj * 0.70)
    n_val = int(n_subj * 0.15)
    train_subj = set(hand_subjects[:n_train])
    val_subj = set(hand_subjects[n_train : n_train + n_val])
    test_subj = set(hand_subjects[n_train + n_val :])

    mask_hand = df["source_dataset"] == "11k_Normal_Hands"
    df.loc[mask_hand & df["patient_id"].isin(train_subj), "split"] = "train"
    df.loc[mask_hand & df["patient_id"].isin(val_subj), "split"] = "val"
    df.loc[mask_hand & df["patient_id"].isin(test_subj), "split"] = "test"

    # Eczema vs Tinea: 70/15/15 split
    ecz_pts = df[df["source_dataset"] == "Eczema_vs_Tinea"]["patient_id"].unique()
    rng.shuffle(ecz_pts)
    n_e = len(ecz_pts)
    n_e_train = int(n_e * 0.70)
    n_e_val = int(n_e * 0.15)
    train_e = set(ecz_pts[:n_e_train])
    val_e = set(ecz_pts[n_e_train : n_e_train + n_e_val])
    test_e = set(ecz_pts[n_e_train + n_e_val :])

    mask_e = df["source_dataset"] == "Eczema_vs_Tinea"
    df.loc[mask_e & df["patient_id"].isin(train_e), "split"] = "train"
    df.loc[mask_e & df["patient_id"].isin(val_e), "split"] = "val"
    df.loc[mask_e & df["patient_id"].isin(test_e), "split"] = "test"

    print("\nFinal Split Distribution:")
    print(pd.crosstab(df["source_dataset"], df["split"]))

    out_csv = "data_normal_disease/clinical_stage_q/manifest_all_local_clinical.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nManifest successfully saved to: {out_csv}")

if __name__ == "__main__":
    main()
