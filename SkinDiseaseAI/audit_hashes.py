import os, sys, time
import numpy as np
import pandas as pd
from PIL import Image

CSV, CACHE, OUT = "outputs/combined_splits.csv", "outputs/audit_hashes.csv", "outputs/audit_near_duplicates.csv"
df = pd.read_csv(CSV)
for c in ("image", "path", "split", "dataset"):
    if c not in df.columns:
        sys.exit(f"combined_splits.csv has no column '{c}'. Columns: {df.columns.tolist()}")
df["image"] = df["image"].astype(str)

def dhash_hex(path):
    try:
        with Image.open(path) as im:
            im.draft("L", (64, 64))
            a = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
        return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes().hex()
    except Exception:
        return ""

cache = {}
if os.path.exists(CACHE):
    c = pd.read_csv(CACHE, dtype=str, keep_default_na=False)
    cache = dict(zip(c["path"], c["hash"]))
todo = [p for p in df["path"] if p not in cache]
print(f"{len(df)} images, {len(todo)} to hash (rest cached). First run can take several minutes.")
t0 = time.time()
for i, p in enumerate(todo, 1):
    cache[p] = dhash_hex(p)
    if i % 2000 == 0:
        print(f"  hashed {i}/{len(todo)}  ({time.time()-t0:.0f}s)")
        pd.DataFrame({"path": list(cache), "hash": list(cache.values())}).to_csv(CACHE, index=False)
pd.DataFrame({"path": list(cache), "hash": list(cache.values())}).to_csv(CACHE, index=False)

df["hash"] = df["path"].map(cache)
bad = int((df["hash"] == "").sum())
print(f"unreadable/failed images: {bad}")
df = df[df["hash"] != ""].reset_index(drop=True)
H = np.frombuffer(bytes.fromhex("".join(df["hash"])), dtype=np.uint8).reshape(len(df), 8)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def nearest(q_idx, r_idx, exclude_self=False):
    R = H[r_idx]; res = []
    for qi in q_idx:
        d = POP[R ^ H[qi]].sum(1).astype(np.int16)
        if exclude_self:
            d[r_idx == qi] = 99
        j = int(d.argmin()); res.append((qi, r_idx[j], int(d[j])))
    return res

rows = []; thresholds = (0, 2, 4, 8)
dxs = df["dx"] if "dx" in df.columns else pd.Series([None] * len(df))
def report(name, q, r, same=False):
    res = nearest(q, r, same)
    dist = np.array([x[2] for x in res])
    print(f"{name:<22}" + "  ".join(f"<= {t}: {int((dist <= t).sum()):>5}" for t in thresholds) + f"   (of {len(q)})")
    for qi, ri, d in res:
        if d <= 8:
            rows.append(dict(query=df.image[qi], query_split=df.split[qi], query_dataset=df.dataset[qi],
                             match=df.image[ri], match_split=df.split[ri], match_dataset=df.dataset[ri],
                             hamming=d, same_label=bool(dxs[qi] == dxs[ri])))
idx = {s: np.where(df.split == s)[0] for s in ("train", "val", "test")}
print("\nHamming distance of each image's nearest neighbour (0 = identical hash; <=4 = almost surely the same photo):")
report("test  -> train", idx["test"], idx["train"])
report("test  -> val", idx["test"], idx["val"])
report("val   -> train", idx["val"], idx["train"])
if "--skip-train-train" not in sys.argv:
    print("checking duplicates inside train (a few minutes)...")
    report("train -> train", idx["train"], idx["train"], same=True)
pd.DataFrame(rows).sort_values("hamming").to_csv(OUT, index=False)
print(f"\nWrote {len(rows)} candidate pairs (distance <= 8) to {OUT}")
print("Rule of thumb: distance 0-4 = same image; 5-8 = look at them yourself. test->train/val matches with distance <= 4 are LEAKAGE.")
