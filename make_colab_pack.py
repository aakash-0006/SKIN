import os, sys, shutil, zipfile
import pandas as pd
# pyrefly: ignore[missing-import]
from PIL import Image

CSV = "outputs/combined_splits.csv"
OUT = r"D:\DOWNLOADS D\colab_pack" if os.name == "nt" else "colab_pack"
SIZE = 320
if len(sys.argv) > 1:
    OUT = sys.argv[1]
df = pd.read_csv(CSV)
for c in ("image", "path", "dx", "lesion_id", "split", "dataset"):
    if c not in df.columns:
        sys.exit(f"{CSV} has no column '{c}'. Columns: {df.columns.tolist()}")
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)

rel, failed = [], []
for i, r in enumerate(df.itertuples(index=False), 1):
    name = f"{r.dataset}__{r.image}.jpg"
    dst = os.path.join(OUT, "images", name)
    rel.append("images/" + name)
    if not os.path.exists(dst):
        try:
            with Image.open(r.path) as im:
                im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(dst, "JPEG", quality=95)
        except Exception as e:
            failed.append((r.path, repr(e)[:80]))
    if i % 2000 == 0:
        print(f"  {i}/{len(df)} images")
if failed:
    print(f"WARNING: {len(failed)} images failed, e.g. {failed[:3]}")
df["relpath"] = rel
df = df[[os.path.exists(os.path.join(OUT, p)) for p in rel]]
df[["image", "relpath", "dx", "lesion_id", "split", "dataset"]].to_csv(os.path.join(OUT, "manifest.csv"), index=False)

for f in ("models/best_model.keras", "models/isic_finetuned.keras", "models/class_indices.json", "outputs/splits.csv",
          "outputs/baseline_test_metrics_KEEP.json"):
    if os.path.exists(f):
        shutil.copy(f, os.path.join(OUT, "models" if f.startswith("models") else "", os.path.basename(f)))
    else:
        print("note: not found, skipped:", f)

zp = OUT.rstrip("\\/") + ".zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    for root, _, files in os.walk(OUT):
        for f in files:
            full = os.path.join(root, f); z.write(full, os.path.relpath(full, OUT))
print(f"\nimages: {len(df)}   split counts: {df.split.value_counts().to_dict()}")
print(f"zip: {zp}  ({os.path.getsize(zp) / 1e9:.2f} GB)\nUpload this zip to Google Drive folder SkinDiseaseAI_colab.")
