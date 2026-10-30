"""Evaluate the gate. Reports false rejection on the frozen TEST split (valid images) and false acceptance on a folder of
unrelated images you provide (sub-folders = categories, e.g. anime, faces, landscapes). Thresholds are never changed here.
Usage: python -m src.eval_gate --unrelated "D:\\path\\to\\unrelated_images"
Tip: decide the --accept-pct BEFORE looking at unrelated-image results, or keep half of the unrelated images unseen."""
import argparse, json, math, os
import numpy as np
import pandas as pd

from src.gate import PROJECT_DIR, check_batch, load_gate, load_path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="outputs/combined_splits.csv")
ap.add_argument("--split", default="test", choices=["test", "val", "validation"], help="split to evaluate (test or val)")
ap.add_argument("--unrelated", default=None)
ap.add_argument("--max-test", type=int, default=0, help="limit images (e.g. 200)")
a = ap.parse_args()
if load_gate() is None:
    raise SystemExit("Gate not fitted. Run: python -m src.fit_gate")

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def run(paths):
    status, reasons = [], []
    for i in range(0, len(paths), 128):
        imgs = [load_path(p) for p in paths[i:i + 128]]
        ok = [im for im in imgs if im is not None]
        verdicts = iter(check_batch(ok))
        for im in imgs:
            if im is None:
                status.append("unreadable"); reasons.append("unreadable")
            else:
                v = next(verdicts); status.append(v["status"]); reasons.append("; ".join(v["reasons"]))
        print(f"  {min(i + 128, len(paths))}/{len(paths)}")
    return np.array(status), reasons

out = {}
df = pd.read_csv(a.csv)
target_splits = ["val", "validation"] if a.split in ("val", "validation") else ["test"]
split_name = "VALIDATION" if a.split in ("val", "validation") else "frozen TEST"
te = df[df.split.isin(target_splits)]
if a.max_test:
    te = te.sample(min(a.max_test, len(te)), random_state=0)
print(f"valid images: {len(te)} from the {split_name} split")
st, rs = run(te.path.tolist())
te = te.assign(gate=st, why=rs)
rej, n = int((st == "rejected").sum()), len(st)
lo, hi = wilson(rej, n)
out[f"valid_{a.split}"] = {"n": n, "false_rejection": rej / n, "ci95": [lo, hi]}
print(f"FALSE REJECTION on valid {split_name} images: {rej}/{n} = {rej / n:.1%} (95% CI {lo:.1%}-{hi:.1%})")
if "dx" in te.columns:
    per = te.groupby("dx").apply(lambda g: pd.Series({"n": len(g), "rejected": int((g.gate == "rejected").sum())}))
    per["false_rejection"] = per.rejected / per.n
    print(per.round(3)); out["valid_test_per_class"] = per.round(4).to_dict("index")

if a.unrelated:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = [os.path.join(r, f) for r, _, fs in os.walk(a.unrelated) for f in fs if f.lower().endswith(exts)]
    cat = [os.path.relpath(os.path.dirname(f), a.unrelated).split(os.sep)[0] for f in files]
    print(f"\nunrelated images: {len(files)}")
    st2, rs2 = run(files)
    u = pd.DataFrame({"file": files, "category": cat, "gate": st2, "why": rs2})
    u = u[u.gate != "unreadable"]
    acc, n2 = int((u.gate == "accepted").sum()), len(u)
    lo, hi = wilson(acc, n2)
    out["unrelated"] = {"n": n2, "false_acceptance": acc / max(n2, 1), "ci95": [lo, hi]}
    print(f"FALSE ACCEPTANCE on unrelated images: {acc}/{n2} = {acc / max(n2, 1):.1%} (95% CI {lo:.1%}-{hi:.1%})")
    for c, g in u.groupby("category"):
        k = int((g.gate == "accepted").sum()); l, h = wilson(k, len(g))
        print(f"  {c:<20} accepted {k}/{len(g)} = {k / len(g):.1%} (95% CI {l:.1%}-{h:.1%})")
        out.setdefault("unrelated_per_category", {})[c] = {"n": len(g), "accepted": k}
    os.makedirs(os.path.join(PROJECT_DIR, "outputs", "validation"), exist_ok=True)
    u.to_csv(os.path.join(PROJECT_DIR, "outputs", "validation", "gate_unrelated_verdicts.csv"), index=False)
save_name = "gate_val_eval.json" if a.split in ("val", "validation") else "gate_eval.json"
save_path = os.path.join(PROJECT_DIR, "outputs", "validation", save_name)
json.dump(out, open(save_path, "w"), indent=2)
print(f"\nsaved outputs/validation/{save_name}. Report only what was measured; with small folders the intervals are wide.")
