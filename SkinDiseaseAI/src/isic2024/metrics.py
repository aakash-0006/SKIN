"""Metrics computation for ISIC 2024 binary malignancy detection."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def compute_pauc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.20) -> float:
    """Compute standardized partial ROC-AUC up to max_fpr (default 0.20).

    Uses sklearn.metrics.roc_auc_score(y_true, y_score, max_fpr=max_fpr).
    Normalized between 0.0 and 1.0; random guessing produces 0.50.
    """
    v_gt = np.asarray(y_true).astype(int)
    v_score = np.asarray(y_score).astype(float).ravel()

    if len(np.unique(v_gt)) < 2:
        return 0.0

    return float(roc_auc_score(v_gt, v_score, max_fpr=max_fpr))


def compute_isic2024_pauc_above_tpr(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = 0.80) -> float:
    """Official Kaggle ISIC 2024 competition evaluation metric.

    Computes partial ROC-AUC for True Positive Rate >= min_tpr (0.80).
    """
    v_gt = np.abs(np.asarray(y_true).astype(int) - 1)
    v_score = 1.0 - np.asarray(y_score).astype(float).ravel()

    if len(np.unique(v_gt)) < 2:
        return 0.0

    return float(roc_auc_score(v_gt, v_score, max_fpr=1.0 - min_tpr))


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    max_fpr: float = 0.20,
) -> dict:
    """Compute full diagnostic and triage metrics for binary malignancy predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_pred_probs).astype(float).ravel()
    y_pred_bin = (y_probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    pauc_20 = compute_pauc(y_true, y_probs, max_fpr=max_fpr)
    pauc_tpr80 = compute_isic2024_pauc_above_tpr(y_true, y_probs, min_tpr=0.80)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))

    # Sensitivity at 80% specificity
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Find operating point where specificity >= 0.80 (i.e. fpr <= 0.20)
    spec_80_idx = np.where(fpr <= 0.20)[0]
    sens_at_80_spec = float(tpr[spec_80_idx[-1]]) if len(spec_80_idx) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "pauc_20": round(pauc_20, 5),
        "pauc_tpr80": round(pauc_tpr80, 5),
        "sensitivity": round(sensitivity, 5),
        "specificity": round(specificity, 5),
        "sensitivity_at_80_spec": round(sens_at_80_spec, 5),
        "precision": round(precision, 5),
        "f1_score": round(f1, 5),
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "sample_counts": {
            "total": len(y_true),
            "positives": int(y_true.sum()),
            "negatives": int((1 - y_true).sum()),
        },
    }
