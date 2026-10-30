import os, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PAIRS, CSV = "outputs/audit_near_duplicates.csv", "outputs/combined_splits.csv"
for f in (PAIRS, CSV):
    if not os.path.exists(f):
        sys.exit(f"Missing {f}. Run audit_hashes.py first.")
pairs = pd.read_csv(PAIRS); comb = pd.read_csv(CSV)
path = dict(zip(comb["image"].astype(str), comb["path"]))
p = pairs[(pairs.query_split == "test") & (pairs.match_split.isin(["train", "val"]))].copy().reset_index(drop=True)
print(len(p), "test -> train/val candidate pairs to verify")
S = 64

def centre(pth):
    with Image.open(pth) as im:
        im = im.convert("RGB")
        w, h = im.size
        return im.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8)))

def vec(im):
    a = np.asarray(im.convert("L").resize((S, S), Image.LANCZOS), dtype=np.float32)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)

def best_corr(a, b):
    best = -1.0
    for k in range(4):
        r = np.rot90(a, k)
        for f in (r, r[:, ::-1]):
            best = max(best, float((f * b).sum()))
    return best

corr = []
for i, r in p.iterrows():
    try:
        corr.append(best_corr(vec(centre(path[str(r["query"])])), vec(centre(path[str(r["match"])]))))
    except Exception:
        corr.append(np.nan)
    if (i + 1) % 200 == 0:
        print(f"  verified {i + 1}/{len(p)}")
p["pixel_corr"] = corr
p = p.sort_values("pixel_corr", ascending=False).reset_index(drop=True)
p.to_csv("outputs/audit_verified_pairs.csv", index=False)

c = p["pixel_corr"].dropna()
print("\nPixel similarity of the hash candidates (1.00 = identical picture):")
for t in (0.99, 0.97, 0.95, 0.90):
    print(f"  correlation >= {t}: {int((c >= t).sum()):>4} of {len(c)}")
print("\nMost similar pairs:")
print(p.head(15)[["query", "match", "match_split", "match_dataset", "hamming", "pixel_corr"]].round(3).to_string(index=False))

top = p.head(12)
W = 170
sheet = Image.new("RGB", (W * 4, (W + 18) * 6), "white"); d = ImageDraw.Draw(sheet)
for k, r in top.iterrows():
    row, col = divmod(k, 2)
    x, y = col * 2 * W, row * (W + 18)
    try:
        for j, key in enumerate(("query", "match")):
            sheet.paste(centre(path[str(r[key])]).resize((W, W)), (x + j * W, y + 18))
        d.text((x + 2, y + 3), f"{r['query']} vs {r['match']}  r={r['pixel_corr']:.2f}", fill="black")
    except Exception:
        pass
sheet.save("outputs/audit_top_pairs.png")
print("\nSaved outputs/audit_verified_pairs.csv and outputs/audit_top_pairs.png (open the picture and look at the pairs yourself).")
print("Rule of thumb: >= 0.97 = very likely the same image (leakage). 0.90-0.97 = look at the picture. Below 0.90 = probably just similar-looking.")
