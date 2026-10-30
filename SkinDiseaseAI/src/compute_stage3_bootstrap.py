"""Compute exact test bootstrap 95% confidence intervals for melanoma recovery.

Evaluates:
- Point estimates and 2000-sample percentile bootstrap 95% CIs for:
  - Melanoma Recall
  - Melanoma Precision
  - Number of False Alarms (False Positives)
  - Macro F1
  - Accuracy
  - One-vs-Rest ROC-AUC
  - PR-AUC
Across:
- Original model (argmax)
- isic_augmented_v1 (argmax)
- isic_augmented_v1 with calibrated threshold (t=0.2327)
- Ensemble (argmax)
- Ensemble with calibrated threshold (t=0.3466)
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.dataset import CLASSES, PROJECT_DIR
from src.melanoma_analysis import apply_mel_threshold

MEL_IDX = CLASSES.index("mel")


def eval_metrics(y_true, probs, t=None):
    if t is None:
        y_pred = probs.argmax(axis=1)
    else:
        y_pred = apply_mel_threshold(probs, t, mel_idx=MEL_IDX)

    y_mel = (y_true == MEL_IDX)
    tp = int(np.sum((y_pred == MEL_IDX) & y_mel))
    fp = int(np.sum((y_pred == MEL_IDX) & (~y_mel)))
    fn = int(np.sum((y_pred != MEL_IDX) & y_mel))
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(np.mean(y_true == y_pred))
    return {"tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": prec, "macro_f1": macro, "accuracy": acc}


def eval_auc(y_true, probs):
    y_mel = (y_true == MEL_IDX).astype(int)
    roc = float(roc_auc_score(y_mel, probs[:, MEL_IDX]))
    pr = float(average_precision_score(y_mel, probs[:, MEL_IDX]))
    return {"roc_auc": roc, "pr_auc": pr}


def main():
    splits_df = pd.read_csv(PROJECT_DIR / "outputs" / "combined_splits.csv")
    class_indices = json.loads((PROJECT_DIR / "models" / "class_indices.json").read_text(encoding="utf-8"))

    test_mask = (splits_df["split"] == "test")
    df_test = splits_df[test_mask].copy().reset_index(drop=True)
    y_test = df_test["dx"].map(class_indices).to_numpy(dtype=np.int32)
    n = len(y_test)

    pred_dir = PROJECT_DIR / "outputs" / "melanoma" / "predictions"
    p_orig_ts = np.load(pred_dir / "orig_test_standard_probs.npy")
    p_aug_ts = np.load(pred_dir / "isic_augmented_v1_test_standard_probs.npy")
    p_ens_ts = 0.5 * (p_orig_ts + p_aug_ts)

    # Thresholds calibrated on HAM-val (1,510 images):
    t_aug_ham = 0.2327
    t_ens_ham = 0.3466

    configs = {
        "original_argmax": (p_orig_ts, None),
        "aug_v1_argmax": (p_aug_ts, None),
        "aug_v1_calibrated": (p_aug_ts, t_aug_ham),
        "ensemble_argmax": (p_ens_ts, None),
        "ensemble_calibrated": (p_ens_ts, t_ens_ham),
    }

    # Non-parametric percentile bootstrap
    np.random.seed(42)
    B = 2000
    boot_idxs = np.random.randint(0, n, size=(B, n))

    results = {}
    print(f"Running {B} bootstrap resamples on test split (N={n})...\n")

    for name, (p_arr, t) in configs.items():
        pt_m = eval_metrics(y_test, p_arr, t)
        pt_auc = eval_auc(y_test, p_arr)

        boot_rec, boot_prec, boot_fp, boot_macro, boot_acc = [], [], [], [], []
        boot_roc, boot_pr = [], []

        for idxs in boot_idxs:
            y_b = y_test[idxs]
            p_b = p_arr[idxs]
            bm = eval_metrics(y_b, p_b, t)
            boot_rec.append(bm["recall"])
            boot_prec.append(bm["precision"])
            boot_fp.append(bm["fp"])
            boot_macro.append(bm["macro_f1"])
            boot_acc.append(bm["accuracy"])

            # AUC bootstrap (ensure at least 1 pos and 1 neg)
            if np.sum(y_b == MEL_IDX) > 0 and np.sum(y_b != MEL_IDX) > 0:
                b_auc = eval_auc(y_b, p_b)
                boot_roc.append(b_auc["roc_auc"])
                boot_pr.append(b_auc["pr_auc"])

        res_entry = {
            "name": name,
            "threshold": t if t is not None else "argmax",
            "recall": round(pt_m["recall"], 4),
            "recall_ci95": [round(float(np.percentile(boot_rec, 2.5)), 4), round(float(np.percentile(boot_rec, 97.5)), 4)],
            "precision": round(pt_m["precision"], 4),
            "precision_ci95": [round(float(np.percentile(boot_prec, 2.5)), 4), round(float(np.percentile(boot_prec, 97.5)), 4)],
            "false_alarms": pt_m["fp"],
            "false_alarms_ci95": [int(np.percentile(boot_fp, 2.5)), int(np.percentile(boot_fp, 97.5))],
            "macro_f1": round(pt_m["macro_f1"], 4),
            "macro_f1_ci95": [round(float(np.percentile(boot_macro, 2.5)), 4), round(float(np.percentile(boot_macro, 97.5)), 4)],
            "accuracy": round(pt_m["accuracy"], 4),
            "accuracy_ci95": [round(float(np.percentile(boot_acc, 2.5)), 4), round(float(np.percentile(boot_acc, 97.5)), 4)],
            "roc_auc": round(pt_auc["roc_auc"], 4),
            "roc_auc_ci95": [round(float(np.percentile(boot_roc, 2.5)), 4), round(float(np.percentile(boot_roc, 97.5)), 4)],
            "pr_auc": round(pt_auc["pr_auc"], 4),
            "pr_auc_ci95": [round(float(np.percentile(boot_pr, 2.5)), 4), round(float(np.percentile(boot_pr, 97.5)), 4)],
            "tp": pt_m["tp"],
            "fn": pt_m["fn"],
        }
        results[name] = res_entry

        print(f"=== {name} (t={res_entry['threshold']}) ===")
        print(f"  Mel Recall:       {res_entry['recall']:.4f} [95% CI: {res_entry['recall_ci95'][0]:.4f} - {res_entry['recall_ci95'][1]:.4f}] ({pt_m['tp']}/187 melanomas)")
        print(f"  Mel Precision:    {res_entry['precision']:.4f} [95% CI: {res_entry['precision_ci95'][0]:.4f} - {res_entry['precision_ci95'][1]:.4f}]")
        print(f"  False Alarms:     {res_entry['false_alarms']} [95% CI: {res_entry['false_alarms_ci95'][0]} - {res_entry['false_alarms_ci95'][1]}] (out of 1307 non-melanomas)")
        print(f"  Macro F1:         {res_entry['macro_f1']:.4f} [95% CI: {res_entry['macro_f1_ci95'][0]:.4f} - {res_entry['macro_f1_ci95'][1]:.4f}]")
        print(f"  Accuracy:         {res_entry['accuracy']:.4f} [95% CI: {res_entry['accuracy_ci95'][0]:.4f} - {res_entry['accuracy_ci95'][1]:.4f}]")
        print(f"  Mel ROC-AUC:      {res_entry['roc_auc']:.4f} [95% CI: {res_entry['roc_auc_ci95'][0]:.4f} - {res_entry['roc_auc_ci95'][1]:.4f}]")
        print(f"  Mel PR-AUC:       {res_entry['pr_auc']:.4f} [95% CI: {res_entry['pr_auc_ci95'][0]:.4f} - {res_entry['pr_auc_ci95'][1]:.4f}]")
        print()

    out_file = PROJECT_DIR / "outputs" / "melanoma" / "stage3_test_bootstrap_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved bootstrap results to {out_file}")


if __name__ == "__main__":
    main()
