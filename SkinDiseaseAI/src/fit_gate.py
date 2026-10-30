"""Fit the input gate. Reads outputs/combined_splits.csv (columns: path, split).
 - embedding distribution: fitted on a sample of TRAIN images
 - thresholds (domain distance + quality limits): chosen on VALIDATION images only. The test split is never used.
Usage: python -m src.fit_gate            (add --weights none only for debugging)"""
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.gate import GATE_PATH, HIGH_KEYS, LOW_KEYS, PROJECT_DIR, embed, load_path, mahalanobis, quality_features, quality_issues

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="outputs/combined_splits.csv")
ap.add_argument("--n-fit", type=int, default=4000)
ap.add_argument("--n-thr", type=int, default=1500)
ap.add_argument("--accept-pct", type=float, default=97.0, help="share of valid validation images the domain check accepts")
ap.add_argument("--quality-pct", type=float, default=0.5, help="percentile used for each quality limit")
ap.add_argument("--weights", default="imagenet")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_csv(a.csv)
for c in ("path", "split"):
    if c not in df.columns:
        raise SystemExit(f"{a.csv} has no column '{c}'")
tr = df[df.split == "train"]; va = df[df.split == "val"]
tr = tr.sample(min(a.n_fit, len(tr)), random_state=a.seed); va = va.sample(min(a.n_thr, len(va)), random_state=a.seed)

def embed_paths(paths, quality=False):
    X, Q, t0 = [], [], time.time()
    for i in range(0, len(paths), 256):
        imgs = [im for im in (load_path(p) for p in paths[i:i + 256]) if im is not None]
        X.append(embed(imgs, a.weights))
        if quality:
            Q += [quality_features(im) for im in imgs]
        print(f"  {min(i + 256, len(paths))}/{len(paths)} images ({time.time() - t0:.0f}s)")
    return np.concatenate(X), Q

print(f"fitting on {len(tr)} TRAIN images, choosing thresholds on {len(va)} VALIDATION images (test split not used)")
Xf, _ = embed_paths(tr.path.tolist())
Xv, Qv = embed_paths(va.path.tolist(), quality=True)
lw = LedoitWolf().fit(Xf)
mu, prec = lw.location_.astype(np.float32), lw.precision_.astype(np.float32)
dv = mahalanobis(Xv, mu, prec)
thr = float(np.percentile(dv, a.accept_pct))
qp = a.quality_pct
lim_low = np.array([np.percentile([q[k] for q in Qv], qp) for k in LOW_KEYS], dtype=np.float32)
lim_high = np.array([np.percentile([q[k] for q in Qv], 100 - qp) for k in HIGH_KEYS], dtype=np.float32)
limits = {**dict(zip(LOW_KEYS, lim_low.tolist())), **dict(zip(HIGH_KEYS, lim_high.tolist()))}
acc = np.mean([(not quality_issues(q, limits)) and d <= thr for q, d in zip(Qv, dv)])
meta = json.dumps({"n_fit": len(tr), "n_thr": len(va), "accept_pct": a.accept_pct, "quality_pct": qp,
                   "seed": a.seed, "weights": a.weights, "fitted_on": "train", "thresholds_chosen_on": "validation"})
np.savez_compressed(GATE_PATH, mu=mu, prec=prec, thr=np.float32(thr), lim_low=lim_low, lim_high=lim_high,
                    weights=np.array(a.weights), meta=np.array(meta))
print(f"\ndomain distance threshold: {thr:.1f}   (validation distance median {np.median(dv):.1f}, max {dv.max():.1f})")
print("quality limits:", {k: round(v, 3) for k, v in limits.items()})
print(f"share of VALIDATION images that pass all checks (by construction, not a test result): {acc:.1%}")
print("saved", GATE_PATH, "\nNext: python -m src.eval_gate --unrelated <folder of non-skin images>")
