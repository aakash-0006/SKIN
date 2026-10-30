import os
import pandas as pd

ddi_dir = r"D:\DOWNLOADS D\ddidiversedermatologyimages"
ddi_csv = os.path.join(ddi_dir, "ddi_metadata.csv")
df_ddi = pd.read_csv(ddi_csv)

norm_rows = []
for idx, r in df_ddi.head(10).iterrows():
    norm_rows.append({
        "image_path": os.path.join(ddi_dir, r["DDI_file"]),
        "dataset_source": "Stanford_DDI",
        "original_label": r["disease"],
        "malignancy_status": "malignant" if r["malignant"] else "benign",
        "binary_label": "UNKNOWN",
        "patient_id": "UNAVAILABLE",
        "skin_tone": f"FST_{r['skin_tone']}",
        "image_domain": "clinical_photography",
        "label_status": "verified_lesion_disease",
        "exclusion_reason": "Pending binary mapping approval (benign lesions != normal skin)"
    })

norm_df = pd.DataFrame(norm_rows)
out_csv = os.path.join("data_normal_disease", "normalized_metadata_sample.csv")
norm_df.to_csv(out_csv, index=False)
print("Normalized metadata sample generated successfully. Rows:", len(norm_df))
