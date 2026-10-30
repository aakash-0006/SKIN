"""Stage 1: In-depth Melanoma (mel) Detection & Decision Threshold Analysis.

Computes:
1. One-vs-rest ROC-AUC and PR-AUC, confusion matrices, and 7-class per-class metrics
   for the original baseline model and isic_augmented_v1 across:
   - HAM10000 validation split (1,510 images)
   - All validation rows (3,440 images)
   - Preserved HAM10000 test split (1,494 images)
2. Melanoma decision threshold tuning on validation (F2-optimal, ~60%, ~70%, ~80% recall),
   evaluated once on test.
3. Test-time augmentation (original + horizontal + vertical flip) and ensemble of
   original + isic_augmented_v1 models.
4. Error breakdown of missed melanomas.

All outputs are saved to outputs/melanoma/.
Baseline models and frozen splits are strictly preserved.
"""
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import tensorflow as tf

from src.dataset import CLASSES, PROJECT_DIR, load_config

MEL_IDX = CLASSES.index("mel")


def make_eval_dataset(paths, img_size, batch_size):
    """Standard evaluation dataset."""
    ds = tf.data.Dataset.from_tensor_slices(list(paths))

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img.set_shape((img_size, img_size, 3))
        return tf.cast(img, tf.float32)

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_or_compute_probs(model_path, tag, split_name, paths, img_size, batch_size, cache_dir, with_tta=False, standard_probs=None):
    """Load cached predictions if available, else compute and cache."""
    cache_file = cache_dir / f"{tag}_{split_name}_{'tta' if with_tta else 'standard'}_probs.npy"
    if cache_file.is_file():
        print(f"Loading cached {tag} ({'TTA' if with_tta else 'standard'}) predictions on {split_name} from {cache_file.name}")
        return np.load(cache_file)

    print(f"Computing {tag} ({'TTA' if with_tta else 'standard'}) predictions on {split_name} ({len(paths)} images)...")
    model = tf.keras.models.load_model(model_path, compile=False)

    if not with_tta:
        ds = make_eval_dataset(paths, img_size, batch_size)
        probs = model.predict(ds, verbose=1)
    else:
        # Re-use standard predictions if provided or cached to save an entire forward pass
        if standard_probs is not None:
            p0 = standard_probs
        else:
            std_file = cache_dir / f"{tag}_{split_name}_standard_probs.npy"
            if std_file.is_file():
                p0 = np.load(std_file)
            else:
                ds = make_eval_dataset(paths, img_size, batch_size)
                p0 = model.predict(ds, verbose=1)

        ds = tf.data.Dataset.from_tensor_slices(list(paths))

        def _load_base(p):
            img = tf.io.read_file(p)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (img_size, img_size))
            img.set_shape((img_size, img_size, 3))
            return tf.cast(img, tf.float32)

        base_ds = ds.map(_load_base, num_parallel_calls=tf.data.AUTOTUNE)
        p1 = model.predict(base_ds.map(lambda x: tf.image.flip_left_right(x), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE), verbose=1)
        p2 = model.predict(base_ds.map(lambda x: tf.image.flip_up_down(x), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE), verbose=1)
        probs = (p0 + p1 + p2) / 3.0

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, probs)
    return probs


def compute_metrics(y_true, y_pred, probs=None):
    """Compute 7-class metrics, macro/weighted F1, and mel ROC/PR-AUC."""
    labels = list(range(len(CLASSES)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class = {}
    for c in CLASSES:
        prec = float(rep[c]["precision"])
        rec = float(rep[c]["recall"])
        f1 = float(rep[c]["f1-score"])
        supp = int(rep[c]["support"])
        f2 = (5.0 * prec * rec) / (4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "support": supp,
        }

    res = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    if probs is not None:
        y_mel = (y_true == MEL_IDX).astype(int)
        p_mel = probs[:, MEL_IDX]
        res["mel_roc_auc"] = round(float(roc_auc_score(y_mel, p_mel)), 4)
        res["mel_pr_auc"] = round(float(average_precision_score(y_mel, p_mel)), 4)

    return res


def apply_mel_threshold(probs, t, mel_idx=MEL_IDX):
    """Rule: Predict mel if P(mel) >= t, otherwise argmax of remaining classes."""
    y_pred = np.empty(len(probs), dtype=np.int32)
    is_mel = probs[:, mel_idx] >= t
    y_pred[is_mel] = mel_idx

    probs_no_mel = probs.copy()
    probs_no_mel[:, mel_idx] = -1.0
    y_pred[~is_mel] = np.argmax(probs_no_mel[~is_mel], axis=1)
    return y_pred


def sweep_thresholds(probs, y_true, t_values=None):
    """Evaluate decision threshold over a range of t values."""
    if t_values is None:
        t_values = np.linspace(0.02, 0.90, 89)

    records = []
    y_mel_true = (y_true == MEL_IDX)
    for t in t_values:
        y_pred = apply_mel_threshold(probs, t)
        rep = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        acc = float((y_true == y_pred).mean())
        prec = float(rep["mel"]["precision"])
        rec = float(rep["mel"]["recall"])
        f1 = float(rep["mel"]["f1-score"])
        f2 = (5.0 * prec * rec) / (4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0

        records.append({
            "threshold": round(float(t), 4),
            "mel_recall": round(rec, 4),
            "mel_precision": round(prec, 4),
            "mel_f1": round(f1, 4),
            "mel_f2": round(f2, 4),
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
        })
    return pd.DataFrame(records)


def analyze_missed_melanomas(y_true, y_pred):
    """Detailed error breakdown of missed melanomas."""
    mel_mask = (y_true == MEL_IDX)
    total_mel = int(mel_mask.sum())
    correct_mel = int((y_pred[mel_mask] == MEL_IDX).sum())
    missed_mel = total_mel - correct_mel
    missed_preds = y_pred[mel_mask & (y_pred != MEL_IDX)]

    breakdown = {}
    for idx, c in enumerate(CLASSES):
        if idx == MEL_IDX:
            continue
        cnt = int((missed_preds == idx).sum())
        pct = (cnt / missed_mel * 100.0) if missed_mel > 0 else 0.0
        breakdown[c] = {"count": cnt, "pct_of_missed": round(pct, 1)}

    return {
        "total_mel": total_mel,
        "correct_mel": correct_mel,
        "missed_mel": missed_mel,
        "mel_recall": round(correct_mel / total_mel if total_mel > 0 else 0.0, 4),
        "mistaken_for": breakdown,
    }


def main():
    cfg = load_config()
    out_dir = PROJECT_DIR / "outputs" / "melanoma"
    cache_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    splits_df = pd.read_csv(PROJECT_DIR / "outputs" / "combined_splits.csv")
    class_indices = json.loads((PROJECT_DIR / "models" / "class_indices.json").read_text(encoding="utf-8"))

    val_all_mask = (splits_df["split"] == "val")
    test_mask = (splits_df["split"] == "test")

    df_val_all = splits_df[val_all_mask].copy().reset_index(drop=True)
    df_test = splits_df[test_mask].copy().reset_index(drop=True)

    ham_in_val = (df_val_all["dataset"] == "HAM10000").to_numpy()
    df_val_ham = df_val_all[ham_in_val].copy().reset_index(drop=True)

    y_val_all = df_val_all["dx"].map(class_indices).to_numpy(dtype=np.int32)
    y_val_ham = y_val_all[ham_in_val]
    y_test = df_test["dx"].map(class_indices).to_numpy(dtype=np.int32)

    print(f"Dataset Counts:")
    print(f"  val_ham: {len(df_val_ham)} images ({int((y_val_ham == MEL_IDX).sum())} melanomas)")
    print(f"  val_all: {len(df_val_all)} images ({int((y_val_all == MEL_IDX).sum())} melanomas)")
    print(f"  test:    {len(df_test)} images ({int((y_test == MEL_IDX).sum())} melanomas)")

    orig_model_path = PROJECT_DIR / "models" / "best_model.keras"
    aug_model_path = PROJECT_DIR / "models" / "isic_augmented_v1.keras"

    # 2. Compute / Load predictions for both models
    img_size = cfg["img_size"]
    batch_size = max(int(cfg.get("batch_size", 32)), 64)

    print("\n--- Obtaining Model Predictions (Standard & TTA) ---")
    # Original model
    p_orig_val_all = get_or_compute_probs(orig_model_path, "orig", "val_all", df_val_all["path"], img_size, batch_size, cache_dir, with_tta=False)
    p_orig_test = get_or_compute_probs(orig_model_path, "orig", "test", df_test["path"], img_size, batch_size, cache_dir, with_tta=False)
    p_orig_val_ham = p_orig_val_all[ham_in_val]

    # Augmented v1 model
    p_aug_val_all = get_or_compute_probs(aug_model_path, "isic_augmented_v1", "val_all", df_val_all["path"], img_size, batch_size, cache_dir, with_tta=False)
    p_aug_test = get_or_compute_probs(aug_model_path, "isic_augmented_v1", "test", df_test["path"], img_size, batch_size, cache_dir, with_tta=False)
    p_aug_val_ham = p_aug_val_all[ham_in_val]

    # TTA predictions (reusing standard_probs for the unaugmented base view)
    p_orig_val_all_tta = get_or_compute_probs(orig_model_path, "orig", "val_all", df_val_all["path"], img_size, batch_size, cache_dir, with_tta=True, standard_probs=p_orig_val_all)
    p_orig_test_tta = get_or_compute_probs(orig_model_path, "orig", "test", df_test["path"], img_size, batch_size, cache_dir, with_tta=True, standard_probs=p_orig_test)
    p_orig_val_ham_tta = p_orig_val_all_tta[ham_in_val]

    p_aug_val_all_tta = get_or_compute_probs(aug_model_path, "isic_augmented_v1", "val_all", df_val_all["path"], img_size, batch_size, cache_dir, with_tta=True, standard_probs=p_aug_val_all)
    p_aug_test_tta = get_or_compute_probs(aug_model_path, "isic_augmented_v1", "test", df_test["path"], img_size, batch_size, cache_dir, with_tta=True, standard_probs=p_aug_test)
    p_aug_val_ham_tta = p_aug_val_all_tta[ham_in_val]

    # Ensemble predictions
    p_ens_val_ham = 0.5 * (p_orig_val_ham + p_aug_val_ham)
    p_ens_val_all = 0.5 * (p_orig_val_all + p_aug_val_all)
    p_ens_test = 0.5 * (p_orig_test + p_aug_test)

    p_ens_val_ham_tta = 0.5 * (p_orig_val_ham_tta + p_aug_val_ham_tta)
    p_ens_val_all_tta = 0.5 * (p_orig_val_all_tta + p_aug_val_all_tta)
    p_ens_test_tta = 0.5 * (p_orig_test_tta + p_aug_test_tta)

    # =========================================================================
    # PART 1: Standard Evaluation (Original vs isic_augmented_v1)
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1 - PART 1: BASELINE & AUGMENTED_V1 EVALUATION (STANDARD ARGMAX)")
    print("="*80)

    part1_results = {
        "original": {
            "val_ham": compute_metrics(y_val_ham, p_orig_val_ham.argmax(axis=1), p_orig_val_ham),
            "val_all": compute_metrics(y_val_all, p_orig_val_all.argmax(axis=1), p_orig_val_all),
            "test": compute_metrics(y_test, p_orig_test.argmax(axis=1), p_orig_test),
        },
        "isic_augmented_v1": {
            "val_ham": compute_metrics(y_val_ham, p_aug_val_ham.argmax(axis=1), p_aug_val_ham),
            "val_all": compute_metrics(y_val_all, p_aug_val_all.argmax(axis=1), p_aug_val_all),
            "test": compute_metrics(y_test, p_aug_test.argmax(axis=1), p_aug_test),
        },
    }

    (out_dir / "stage1_part1_baseline_metrics.json").write_text(json.dumps(part1_results, indent=2), encoding="utf-8")

    # =========================================================================
    # PART 2: Melanoma Decision Threshold Tuning
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1 - PART 2: MELANOMA DECISION THRESHOLD TUNING (ON VALIDATION)")
    print("="*80)

    # Threshold sweeps on validation for isic_augmented_v1
    sweep_val_ham = sweep_thresholds(p_aug_val_ham, y_val_ham)
    sweep_val_all = sweep_thresholds(p_aug_val_all, y_val_all)
    sweep_val_ham.to_csv(out_dir / "threshold_sweep_aug_val_ham.csv", index=False)
    sweep_val_all.to_csv(out_dir / "threshold_sweep_aug_val_all.csv", index=False)

    # Threshold sweeps on validation for original model
    sweep_orig_val_ham = sweep_thresholds(p_orig_val_ham, y_val_ham)
    sweep_orig_val_all = sweep_thresholds(p_orig_val_all, y_val_all)
    sweep_orig_val_ham.to_csv(out_dir / "threshold_sweep_orig_val_ham.csv", index=False)
    sweep_orig_val_all.to_csv(out_dir / "threshold_sweep_orig_val_all.csv", index=False)

    # Select criteria on val_all (and val_ham)
    # Let's inspect selections for isic_augmented_v1 on val_all
    idx_f2_val_all = sweep_val_all["mel_f2"].idxmax()
    t_f2 = sweep_val_all.loc[idx_f2_val_all, "threshold"]

    # Target ~60%, ~70%, ~80% mel recall on val_all
    idx_60 = (sweep_val_all["mel_recall"] - 0.60).abs().idxmin()
    t_60 = sweep_val_all.loc[idx_60, "threshold"]

    idx_70 = (sweep_val_all["mel_recall"] - 0.70).abs().idxmin()
    t_70 = sweep_val_all.loc[idx_70, "threshold"]

    idx_80 = (sweep_val_all["mel_recall"] - 0.80).abs().idxmin()
    t_80 = sweep_val_all.loc[idx_80, "threshold"]

    selected_thresholds = {
        "Default argmax (implied ~0.5)": None,
        f"F2-optimal (t={t_f2:.2f})": t_f2,
        f"~60% Recall target (t={t_60:.2f})": t_60,
        f"~70% Recall target (t={t_70:.2f})": t_70,
        f"~80% Recall target (t={t_80:.2f})": t_80,
    }

    # Evaluate selected thresholds on val_ham, val_all, and test ONCE
    thresh_table_aug = []
    for label, t in selected_thresholds.items():
        row = {"Criterion": label, "Threshold": f"{t:.2f}" if t is not None else "argmax"}
        for split_key, p_arr, y_arr in [("val_ham", p_aug_val_ham, y_val_ham),
                                         ("val_all", p_aug_val_all, y_val_all),
                                         ("test", p_aug_test, y_test)]:
            if t is None:
                y_pred = p_arr.argmax(axis=1)
            else:
                y_pred = apply_mel_threshold(p_arr, t)
            m = compute_metrics(y_arr, y_pred)
            row[f"{split_key}_mel_rec"] = m["per_class"]["mel"]["recall"]
            row[f"{split_key}_mel_prec"] = m["per_class"]["mel"]["precision"]
            row[f"{split_key}_macro_f1"] = m["macro_f1"]
            row[f"{split_key}_acc"] = m["accuracy"]
        thresh_table_aug.append(row)

    df_thresh_aug = pd.DataFrame(thresh_table_aug)
    df_thresh_aug.to_csv(out_dir / "stage1_part2_threshold_evaluation_augmented.csv", index=False)

    # =========================================================================
    # PART 3: Ensembles & Test-Time Augmentation (TTA)
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1 - PART 3: ENSEMBLE & TTA EVALUATION")
    print("="*80)

    variants = {
        "Original (Standard)": (p_orig_val_ham, p_orig_val_all, p_orig_test),
        "Original + TTA": (p_orig_val_ham_tta, p_orig_val_all_tta, p_orig_test_tta),
        "ISIC Aug v1 (Standard)": (p_aug_val_ham, p_aug_val_all, p_aug_test),
        "ISIC Aug v1 + TTA": (p_aug_val_ham_tta, p_aug_val_all_tta, p_aug_test_tta),
        "Ensemble (Orig + Aug v1)": (p_ens_val_ham, p_ens_val_all, p_ens_test),
        "Ensemble + TTA": (p_ens_val_ham_tta, p_ens_val_all_tta, p_ens_test_tta),
    }

    part3_records = []
    for var_name, (p_vh, p_va, p_ts) in variants.items():
        # Evaluate standard argmax
        m_vh = compute_metrics(y_val_ham, p_vh.argmax(axis=1), p_vh)
        m_va = compute_metrics(y_val_all, p_va.argmax(axis=1), p_va)
        m_ts = compute_metrics(y_test, p_ts.argmax(axis=1), p_ts)

        part3_records.append({
            "Variant": var_name,
            # Val HAM
            "val_ham_acc": m_vh["accuracy"],
            "val_ham_macro_f1": m_vh["macro_f1"],
            "val_ham_mel_rec": m_vh["per_class"]["mel"]["recall"],
            "val_ham_mel_prec": m_vh["per_class"]["mel"]["precision"],
            "val_ham_mel_roc_auc": m_vh["mel_roc_auc"],
            "val_ham_mel_pr_auc": m_vh["mel_pr_auc"],
            # Val ALL
            "val_all_acc": m_va["accuracy"],
            "val_all_macro_f1": m_va["macro_f1"],
            "val_all_mel_rec": m_va["per_class"]["mel"]["recall"],
            "val_all_mel_prec": m_va["per_class"]["mel"]["precision"],
            "val_all_mel_roc_auc": m_va["mel_roc_auc"],
            "val_all_mel_pr_auc": m_va["mel_pr_auc"],
            # Test
            "test_acc": m_ts["accuracy"],
            "test_macro_f1": m_ts["macro_f1"],
            "test_mel_rec": m_ts["per_class"]["mel"]["recall"],
            "test_mel_prec": m_ts["per_class"]["mel"]["precision"],
            "test_mel_roc_auc": m_ts["mel_roc_auc"],
            "test_mel_pr_auc": m_ts["mel_pr_auc"],
        })

    df_part3 = pd.DataFrame(part3_records)
    df_part3.to_csv(out_dir / "stage1_part3_ensemble_tta_comparison.csv", index=False)

    # Also evaluate thresholding on the best Ensemble/TTA on validation!
    # Sweep on Ensemble (Standard and TTA)
    sweep_ens_val_all = sweep_thresholds(p_ens_val_all, y_val_all)
    idx_ens_f2 = sweep_ens_val_all["mel_f2"].idxmax()
    t_ens_f2 = sweep_ens_val_all.loc[idx_ens_f2, "threshold"]

    # =========================================================================
    # PART 4: Missed Melanoma Error Breakdown
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1 - PART 4: MISSED MELANOMA ERROR ANALYSIS")
    print("="*80)

    missed_analysis = {
        "original": {
            "val_ham": analyze_missed_melanomas(y_val_ham, p_orig_val_ham.argmax(axis=1)),
            "val_all": analyze_missed_melanomas(y_val_all, p_orig_val_all.argmax(axis=1)),
            "test": analyze_missed_melanomas(y_test, p_orig_test.argmax(axis=1)),
        },
        "isic_augmented_v1": {
            "val_ham": analyze_missed_melanomas(y_val_ham, p_aug_val_ham.argmax(axis=1)),
            "val_all": analyze_missed_melanomas(y_val_all, p_aug_val_all.argmax(axis=1)),
            "test": analyze_missed_melanomas(y_test, p_aug_test.argmax(axis=1)),
        },
        "isic_augmented_v1_f2_threshold": {
            "val_ham": analyze_missed_melanomas(y_val_ham, apply_mel_threshold(p_aug_val_ham, t_f2)),
            "val_all": analyze_missed_melanomas(y_val_all, apply_mel_threshold(p_aug_val_all, t_f2)),
            "test": analyze_missed_melanomas(y_test, apply_mel_threshold(p_aug_test, t_f2)),
        },
        "ensemble_standard": {
            "val_ham": analyze_missed_melanomas(y_val_ham, p_ens_val_ham.argmax(axis=1)),
            "val_all": analyze_missed_melanomas(y_val_all, p_ens_val_all.argmax(axis=1)),
            "test": analyze_missed_melanomas(y_test, p_ens_test.argmax(axis=1)),
        },
    }

    (out_dir / "stage1_part4_missed_melanoma_analysis.json").write_text(json.dumps(missed_analysis, indent=2), encoding="utf-8")

    print(f"\nAll Stage 1 analyses successfully computed and saved to {out_dir}!")


if __name__ == "__main__":
    main()
