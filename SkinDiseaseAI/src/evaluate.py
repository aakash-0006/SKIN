"""Compare the original and ISIC fine-tuned models on the preserved HAM test split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.dataset import CLASSES, PROJECT_DIR, load_config, resolve_split_paths
from src.preprocessing import make_dataset


def evaluate_model(model_path, test, class_to_idx, cfg, out_dir, tag):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"\nEvaluating {tag} checkpoint: {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    expected = (cfg["img_size"], cfg["img_size"], 3)
    if tuple(model.input_shape[1:]) != expected or model.output_shape[-1] != len(CLASSES):
        raise ValueError(f"Incompatible model {model_path}: input={model.input_shape}, output={model.output_shape}")
    ds = make_dataset(test.path, test.y, cfg["img_size"], cfg["batch_size"])
    probs = model.predict(ds, verbose=1)
    y_true, y_pred = test.y.to_numpy(), probs.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_dir / f"{tag}_classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / f"{tag}_confusion_matrix.csv")
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{tag}: confusion matrix on preserved HAM test set")
    plt.tight_layout(); plt.savefig(out_dir / f"{tag}_confusion_matrix.png", dpi=160); plt.close()
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{tag}: normalized confusion matrix")
    plt.tight_layout(); plt.savefig(out_dir / f"{tag}_confusion_matrix_normalized.png", dpi=160); plt.close()
    min_support = int(cfg.get("min_reliable_support", 20))
    class_counts = test.dx.value_counts().reindex(CLASSES, fill_value=0)
    per_class = {}
    for name in CLASSES:
        support = int(class_counts[name])
        per_class[name] = {
            "test_count": support,
            "precision": float(rep[name]["precision"]),
            "recall": float(rep[name]["recall"]),
            "f1": float(rep[name]["f1-score"]),
            "metrics_reliability": "low support; interpret cautiously" if support < min_support else "support threshold met",
        }
    metrics = {
        "model": str(model_path), "test_images": int(len(test)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "test_counts_by_class": {c: int(class_counts[c]) for c in CLASSES},
        "low_support_threshold": min_support,
        "interpretation": "Metrics for classes below the support threshold (df, vasc) are unstable and should be interpreted cautiously.",
    }
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-model", default=cfg.get("model_path", "models/best_model.keras"))
    parser.add_argument("--isic-model", default=cfg.get("new_model_path", "models/isic_finetuned.keras"))
    parser.add_argument("--augmented-model", default="models/isic_augmented_v1.keras")
    parser.add_argument("--skip-original", action="store_true", help="Skip evaluating original baseline")
    parser.add_argument("--skip-isic", action="store_true", help="Skip evaluating isic_finetuned model")
    args = parser.parse_args()

    class_to_idx = json.loads((PROJECT_DIR / cfg["class_indices_path"]).read_text(encoding="utf-8"))
    expected = {c: i for i, c in enumerate(CLASSES)}
    if class_to_idx != expected:
        raise ValueError(f"Class order mismatch: {class_to_idx}; expected {expected}")

    splits_path = PROJECT_DIR / cfg.get("existing_splits_path", "outputs/splits.csv")
    splits = resolve_split_paths(pd.read_csv(splits_path), cfg["dataset_root"])
    test = splits[splits.split == "test"].copy()
    test["y"] = test.dx.map(class_to_idx)
    if test.empty or test.y.isna().any():
        raise ValueError("Preserved test split is empty or contains labels outside the seven classes.")
    test["y"] = test.y.astype("int32")

    missing = [p for p in test.path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} preserved test images are missing; sample: {missing[:5]}")

    out = PROJECT_DIR / "outputs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    orig_path = PROJECT_DIR / args.original_model
    if not args.skip_original and orig_path.is_file():
        results["original"] = evaluate_model(orig_path, test, class_to_idx, cfg, out, "original")
    elif not orig_path.is_file():
        print(f"Original model not found at {orig_path}")

    isic_path = PROJECT_DIR / args.isic_model
    if not args.skip_isic and isic_path.is_file():
        results["isic_finetuned"] = evaluate_model(isic_path, test, class_to_idx, cfg, out, "isic_finetuned")
    elif not isic_path.is_file():
        print(f"ISIC finetuned model not found at {isic_path}")

    aug_path = PROJECT_DIR / args.augmented_model
    if aug_path.is_file():
        results["isic_augmented_v1"] = evaluate_model(aug_path, test, class_to_idx, cfg, out, "isic_augmented_v1")
    else:
        print(f"Augmented model not found at {aug_path}. (Train with `python -m src.train` first)")

    comparison_report = {
        "test_manifest": str(splits_path),
        "test_images": int(len(test)),
        "models": {k: {"path": str(results[k]["model"]),
                       "accuracy": results[k]["accuracy"],
                       "macro_f1": results[k]["macro_f1"],
                       "weighted_f1": results[k]["weighted_f1"],
                       "mel_recall": results[k]["per_class"]["mel"]["recall"],
                       "mel_precision": results[k]["per_class"]["mel"]["precision"]}
                   for k in results},
    }

    if "original" in results and "isic_augmented_v1" in results:
        aug_m = results["isic_augmented_v1"]
        base_m = results["original"]
        comparison_report["augmented_vs_original"] = {
            "accuracy_delta": aug_m["accuracy"] - base_m["accuracy"],
            "macro_f1_delta": aug_m["macro_f1"] - base_m["macro_f1"],
            "weighted_f1_delta": aug_m["weighted_f1"] - base_m["weighted_f1"],
            "mel_recall_delta": aug_m["per_class"]["mel"]["recall"] - base_m["per_class"]["mel"]["recall"],
            "mel_precision_delta": aug_m["per_class"]["mel"]["precision"] - base_m["per_class"]["mel"]["precision"],
            "per_class_f1_delta": {c: aug_m["per_class"][c]["f1"] - base_m["per_class"][c]["f1"] for c in CLASSES},
        }

    if "isic_finetuned" in results and "isic_augmented_v1" in results:
        aug_m = results["isic_augmented_v1"]
        isic_m = results["isic_finetuned"]
        comparison_report["augmented_vs_isic_finetuned"] = {
            "accuracy_delta": aug_m["accuracy"] - isic_m["accuracy"],
            "macro_f1_delta": aug_m["macro_f1"] - isic_m["macro_f1"],
            "weighted_f1_delta": aug_m["weighted_f1"] - isic_m["weighted_f1"],
            "mel_recall_delta": aug_m["per_class"]["mel"]["recall"] - isic_m["per_class"]["mel"]["recall"],
            "mel_precision_delta": aug_m["per_class"]["mel"]["precision"] - isic_m["per_class"]["mel"]["precision"],
            "per_class_f1_delta": {c: aug_m["per_class"][c]["f1"] - isic_m["per_class"][c]["f1"] for c in CLASSES},
        }

    three_way_path = out / "three_model_comparison.json"
    three_way_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    print(f"\nThree-model comparison written to: {three_way_path}")

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Metric':<25} | {'Original Baseline':<18} | {'ISIC Fine-Tuned':<18} | {'ISIC Augmented v1':<18}")
    print("-" * 85)
    for metric_name, key in [("Test Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")]:
        v0 = f"{results['original'][key]:.4f}" if "original" in results else "N/A"
        v1 = f"{results['isic_finetuned'][key]:.4f}" if "isic_finetuned" in results else "N/A"
        v2 = f"{results['isic_augmented_v1'][key]:.4f}" if "isic_augmented_v1" in results else "Pending"
        print(f"{metric_name:<25} | {v0:<18} | {v1:<18} | {v2:<18}")

    print("-" * 85)
    print("Per-Class F1 Scores:")
    for c in CLASSES:
        v0 = f"{results['original']['per_class'][c]['f1']:.4f}" if "original" in results else "N/A"
        v1 = f"{results['isic_finetuned']['per_class'][c]['f1']:.4f}" if "isic_finetuned" in results else "N/A"
        v2 = f"{results['isic_augmented_v1']['per_class'][c]['f1']:.4f}" if "isic_augmented_v1" in results else "Pending"
        print(f"  {c:<23} | {v0:<18} | {v1:<18} | {v2:<18}")
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
