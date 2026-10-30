"""Input gate (EXPERIMENTAL research prototype - not a medical device).

Runs BEFORE any disease classifier and decides whether an image looks like the dermoscopic lesion images the
models were trained on. If not, no disease label should be shown. It combines:
  1. file checks (empty / corrupt / unsupported / too small),
  2. image-quality limits learned from valid VALIDATION images (darkness, contrast, blur, glare),
  3. a domain check: Mahalanobis distance of a frozen ImageNet EfficientNetB0 embedding from the embeddings of
     TRAIN images (fitted by src/fit_gate.py, threshold chosen on VALIDATION images, never on the test set).
Passing the gate does NOT mean an image is a valid lesion photo, healthy, or diagnosable.
"""
import io
import os
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_PATH = os.path.join(PROJECT_DIR, "models", "ood_gate.npz")
MIN_SIDE, MAX_BYTES = 64, 15 * 1024 * 1024
LOW_KEYS = ("lum", "contrast", "blur")      # image is rejected when BELOW the limit
HIGH_KEYS = ("dark_frac", "sat_frac")       # image is rejected when ABOVE the limit
QUALITY_TEXT = {
    "lum": "the image is too dark", "contrast": "the image has too little contrast / visible detail",
    "blur": "the image is too blurry or lacks fine detail", "dark_frac": "too much of the image is nearly black",
    "sat_frac": "the image has glare or is overexposed",
}


# ---------- file checks ----------
def read_image(data):
    """bytes -> (PIL RGB image, None) or (None, error message)."""
    if not data:
        return None, "The file is empty."
    if len(data) > MAX_BYTES:
        return None, "The file is larger than 15 MB."
    try:
        im = Image.open(io.BytesIO(data)); im.verify()
        im = Image.open(io.BytesIO(data)); im.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None, "The file is not a readable image."
    if im.format not in ("JPEG", "PNG"):
        return None, f"Unsupported format ({im.format}). Use JPG or PNG."
    if min(im.size) < MIN_SIDE:
        return None, f"The image is too small ({im.size[0]}x{im.size[1]} px)."
    return im.convert("RGB"), None


def load_path(path):
    """Fast loader for dataset files; returns None if unreadable."""
    try:
        with Image.open(path) as im:
            im.draft("RGB", (448, 448))
            return im.convert("RGB")
    except Exception:
        return None


# ---------- quality ----------
def quality_features(img):
    g = np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return {"lum": float(g.mean()), "contrast": float(g.std()), "blur": float(lap.var()),
            "dark_frac": float((g < 20).mean()), "sat_frac": float((g >= 245).mean())}


def quality_issues(feats, limits):
    """limits: {'lum': lower, ..., 'dark_frac': upper, ...}. Returns a list of human-readable problems."""
    out = [k for k in LOW_KEYS if k in limits and feats[k] < limits[k]]
    out += [k for k in HIGH_KEYS if k in limits and feats[k] > limits[k]]
    return out


# ---------- domain (out-of-distribution) check ----------
@lru_cache(maxsize=2)
def _embedder(weights):
    try:
        import keras
        if weights == "none":
            keras.utils.set_random_seed(0)
        return keras.applications.EfficientNetB0(include_top=False, weights=None if weights == "none" else weights,
                                                 pooling="avg", input_shape=(224, 224, 3))
    except Exception:
        import tensorflow as tf
        if weights == "none":
            tf.keras.utils.set_random_seed(0)
        return tf.keras.applications.EfficientNetB0(include_top=False, weights=None if weights == "none" else weights,
                                                    pooling="avg", input_shape=(224, 224, 3))


def embed(imgs, weights="imagenet", batch=32):
    """List of PIL images -> (n, 1280) embeddings. Raw 0-255 pixels (EfficientNet normalizes internally)."""
    model = _embedder(weights)
    out = []
    for i in range(0, len(imgs), batch):
        arr = np.stack([np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32)
                        for im in imgs[i:i + batch]])
        out.append(model.predict(arr, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, 1280), dtype=np.float32)


def mahalanobis(x, mu, prec):
    d = x - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


@lru_cache(maxsize=1)
def load_gate():
    if not os.path.exists(GATE_PATH):
        return None
    z = np.load(GATE_PATH, allow_pickle=False)
    limits = {k: float(v) for k, v in zip(LOW_KEYS, z["lim_low"])}
    limits.update({k: float(v) for k, v in zip(HIGH_KEYS, z["lim_high"])})
    return {"mu": z["mu"], "prec": z["prec"], "thr": float(z["thr"]), "limits": limits,
            "weights": str(z["weights"]), "meta": str(z["meta"])}


def check_batch(imgs, gate=None):
    """imgs: list of PIL RGB images -> list of verdict dicts (status accepted/rejected/unavailable)."""
    gate = gate or load_gate()
    if gate is None:
        return [{"status": "unavailable", "reasons": ["The input gate has not been fitted (run python -m src.fit_gate)."],
                 "message": "Input validation is not set up yet, so this result is NOT validated.", "details": {}}
                for _ in imgs]
    dist = mahalanobis(embed(imgs, gate["weights"]), gate["mu"], gate["prec"])
    res = []
    for im, d in zip(imgs, dist):
        feats = quality_features(im)
        reasons = [QUALITY_TEXT[k] for k in quality_issues(feats, gate["limits"])]
        if d > gate["thr"]:
            reasons.append("the image does not look like the dermoscopic lesion images this model was trained on")
        status = "rejected" if reasons else "accepted"
        msg = ("No prediction is shown because " + "; ".join(reasons) + "." if reasons else
               "Image passed the input checks. This does not mean it is a valid lesion photo or that the result is correct.")
        res.append({"status": status, "reasons": reasons, "message": msg,
                    "details": {"distance": float(d), "threshold": gate["thr"], "quality": feats}})
    return res


def check_image(data):
    """bytes or PIL image -> verdict dict. Never raises for bad input."""
    if isinstance(data, (bytes, bytearray)):
        img, err = read_image(bytes(data))
        if err:
            return {"status": "rejected", "reasons": [err], "message": "No prediction is shown. " + err, "details": {}}
    else:
        img = data.convert("RGB")
    return check_batch([img])[0]
