"""Isolated Experiment Script: EXP-ENS-001 (Effective Number of Samples Beta=0.999).

Evaluates ENS class weighting on the existing EfficientNetB0 skin-lesion classifier.
Protects all existing model checkpoints, frozen splits, and configs.
Evaluates ONLY on the validation split. Does NOT evaluate on the frozen test set.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import sys
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.dataset import CLASSES, load_config
from src.preprocessing import apply_augmentation_to_model, make_dataset

EXPERIMENT_ID = "EXP-ENS-001"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "experiments" / EXPERIMENT_ID
CHECKPOINT_PATH = PROJECT_DIR / "models" / "experiments" / "isic_ens_beta999.keras"
SPLITS_CSV = PROJECT_DIR / "outputs" / "combined_splits.csv"
INIT_MODEL_PATH = PROJECT_DIR / "models" / "isic_finetuned.keras"

PROTECTED_PATHS = [
    PROJECT_DIR / "models" / "best_model.keras",
    PROJECT_DIR / "models" / "isic_finetuned.keras",
    PROJECT_DIR / "models" / "isic_augmented_v1.keras",
    PROJECT_DIR / "models" / "ood_gate.npz",
    PROJECT_DIR / "outputs" / "splits.csv",
    PROJECT_DIR / "outputs" / "combined_splits.csv",
    PROJECT_DIR / "config.json",
    PROJECT_DIR / "app.py",
]


def check_protection():
    """Ensure no target output collides with protected assets."""
    target_paths = [CHECKPOINT_PATH, OUTPUT_DIR]
    for target in target_paths:
        for prot in PROTECTED_PATHS:
            if target.resolve() == prot.resolve():
                raise RuntimeError(f"PROTECTION VIOLATION: Refusing to target protected file: {target}")


def compute_ens_weights(y_train, classes=CLASSES, beta=0.999):
    """Compute Effective Number of Samples (ENS) weights strictly from training data.
    
    Formula:
        E_n = (1 - beta^n) / (1 - beta)
        w_c = 1 / E_n
    Normalized so that the mean class weight across classes is 1.0 (sum w_c = K).
    """
    K = len(classes)
    y_arr = np.asarray(y_train, dtype=int)
    counts = np.bincount(y_arr, minlength=K)
    total = len(y_arr)
    
    raw_weights = np.zeros(K, dtype=float)
    eff_nums = np.zeros(K, dtype=float)
    for c in range(K):
        n_c = counts[c]
        if n_c > 0:
            eff_n = (1.0 - np.power(beta, n_c)) / (1.0 - beta)
            eff_nums[c] = eff_n
            raw_weights[c] = 1.0 / eff_n
        else:
            eff_nums[c] = 1.0
            raw_weights[c] = 1.0

    # Normalize so average class weight is 1.0 (sum of weights = K)
    norm_weights = K * (raw_weights / np.sum(raw_weights))
    
    weight_dict = {int(i): float(norm_weights[i]) for i in range(K)}
    
    table_data = []
    nv_idx = classes.index("nv") if "nv" in classes else 0
    nv_w = norm_weights[nv_idx]
    
    for i, name in enumerate(classes):
        table_data.append({
            "class_name": name,
            "class_index": i,
            "train_count": int(counts[i]),
            "train_fraction": round(float(counts[i] / total), 6),
            "effective_num_samples": round(float(eff_nums[i]), 4),
            "raw_weight": round(float(raw_weights[i]), 8),
            "normalized_weight": round(float(norm_weights[i]), 6),
            "ratio_vs_nv": round(float(norm_weights[i] / nv_w), 4)
        })
        
    return weight_dict, table_data


def evaluate_split(model, df_subset, cfg, split_name):
    """Evaluate model on a specific validation subset without touching test data."""
    ds = make_dataset(df_subset.path, df_subset.y, cfg["img_size"], cfg["batch_size"])
    probs = model.predict(ds, verbose=0)
    y_true = df_subset.y.to_numpy()
    y_pred = probs.argmax(axis=1)
    
    K = len(CLASSES)
    labels = list(range(K))
    y_onehot = np.eye(K)[y_true]
    
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    rep = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Per-class metrics
    per_class = {}
    for i, c in enumerate(CLASSES):
        roc = float(roc_auc_score(y_onehot[:, i], probs[:, i])) if np.sum(y_onehot[:, i]) > 0 else 0.0
        pr = float(average_precision_score(y_onehot[:, i], probs[:, i])) if np.sum(y_onehot[:, i]) > 0 else 0.0
        tn = float(np.sum((y_true != i) & (y_pred != i)))
        fp = float(np.sum((y_true != i) & (y_pred == i)))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class[c] = {
            "support": int(rep[c]["support"]),
            "precision": round(float(rep[c]["precision"]), 4),
            "recall": round(float(rep[c]["recall"]), 4),
            "f1": round(float(rep[c]["f1-score"]), 4),
            "specificity": round(spec, 4),
            "roc_auc_ovr": round(roc, 4),
            "pr_auc_ovr": round(pr, 4)
        }
        
    mel_idx = CLASSES.index("mel")
    non_mel_mask = (y_true != mel_idx)
    non_mel_total = int(np.sum(non_mel_mask))
    mel_false_alarms = int(np.sum((y_pred == mel_idx) & non_mel_mask))
    mel_far = mel_false_alarms / non_mel_total if non_mel_total > 0 else 0.0
    
    res = {
        "split_name": split_name,
        "sample_count": int(len(df_subset)),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_roc_auc": round(float(np.mean([per_class[c]["roc_auc_ovr"] for c in CLASSES])), 4),
        "macro_pr_auc": round(float(np.mean([per_class[c]["pr_auc_ovr"] for c in CLASSES])), 4),
        "melanoma_metrics": {
            "support": int(per_class["mel"]["support"]),
            "recall_sensitivity": per_class["mel"]["recall"],
            "precision": per_class["mel"]["precision"],
            "f1": per_class["mel"]["f1"],
            "roc_auc_ovr": per_class["mel"]["roc_auc_ovr"],
            "pr_auc_ovr": per_class["mel"]["pr_auc_ovr"],
            "false_alarms_count": mel_false_alarms,
            "non_melanoma_total": non_mel_total,
            "false_alarm_rate": round(mel_far, 4)
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }
    return res, probs, cm


def run_experiment():
    check_protection()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    print(f"=== Starting Experiment: {EXPERIMENT_ID} ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random seed: {seed}")
    
    # 1. Load data
    print(f"Loading split manifest: {SPLITS_CSV}")
    combined = pd.read_csv(SPLITS_CSV)
    
    # Strictly isolate train and val
    train = combined[combined.split == "train"].copy()
    val = combined[combined.split == "val"].copy()
    
    print(f"Dataset partitions loaded: Train={len(train)}, Val={len(val)}")
    print("NOTE: Frozen test split is strictly excluded from training and validation evaluation.")
    
    # 2. Compute ENS Beta=0.999 weights from TRAIN set only
    class_weights, weights_table = compute_ens_weights(train.y.values, classes=CLASSES, beta=0.999)
    
    df_weights = pd.DataFrame(weights_table)
    df_weights.to_csv(OUTPUT_DIR / "class_weights.csv", index=False)
    with open(OUTPUT_DIR / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump({"weights": class_weights, "details": weights_table}, f, indent=2)
        
    print("\n--- ENS (Beta=0.999) Class Weights (Train Split Only) ---")
    print(df_weights[["class_name", "train_count", "normalized_weight", "ratio_vs_nv"]].to_string(index=False))
    print("---------------------------------------------------------\n")
    
    # 3. Create tf.data datasets
    batch_size = cfg.get("batch_size", 32)
    img_size = cfg.get("img_size", 224)
    train_ds = make_dataset(train.path, train.y, img_size, batch_size, training=True, seed=seed)
    val_ds = make_dataset(val.path, val.y, img_size, batch_size, training=False)
    
    # 4. Load base model and apply augmentation layer
    print(f"Loading initial model checkpoint: {INIT_MODEL_PATH}")
    if not INIT_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Starting model checkpoint not found: {INIT_MODEL_PATH}")
    source_model = tf.keras.models.load_model(INIT_MODEL_PATH, compile=False)
    
    print("Applying conservative online image augmentation layer...")
    model = apply_augmentation_to_model(source_model)
    
    # Fine-tune top 40 layers of EfficientNetB0
    finetune_layers = cfg.get("finetune_layers", 40)
    lr = cfg.get("finetune_lr", 1e-5)
    epochs = cfg.get("finetune_epochs", 10)
    
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    frozen_prefix = max(0, len(base.layers) - finetune_layers)
    for layer in base.layers[:frozen_prefix]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    print(f"EfficientNetB0 configured: {len(base.layers)} total layers, top {finetune_layers} trainable (BatchNormalization frozen).")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # 5. Callbacks & Training
    history_csv = OUTPUT_DIR / "training_history.csv"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
    ]
    
    print(f"Starting training for up to {epochs} epochs (Batch size={batch_size}, LR={lr})...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds ({training_duration/60.0:.2f} minutes).")
    
    # 6. Load best restored checkpoint and evaluate on validation subsets
    print(f"\nLoading best checkpoint for validation evaluation: {CHECKPOINT_PATH}")
    best_model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    
    # Subset 1: Full validation partition (val_all, N=3,440)
    val_all_metrics, val_all_probs, val_all_cm = evaluate_split(best_model, val, cfg, "val_all")
    
    # Subset 2: HAM-only validation partition (val_ham, N=1,510)
    val_ham_df = val[val.dataset == "HAM10000"].copy()
    val_ham_metrics, val_ham_probs, val_ham_cm = evaluate_split(best_model, val_ham_df, cfg, "val_ham")
    
    # Save validation confusion matrices and predictions
    np.save(OUTPUT_DIR / "val_all_probs.npy", val_all_probs)
    np.save(OUTPUT_DIR / "val_ham_probs.npy", val_ham_probs)
    
    pd.DataFrame(val_all_cm, index=CLASSES, columns=CLASSES).to_csv(OUTPUT_DIR / "val_all_confusion_matrix.csv")
    pd.DataFrame(val_ham_cm, index=CLASSES, columns=CLASSES).to_csv(OUTPUT_DIR / "val_ham_confusion_matrix.csv")
    
    val_losses = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else len(history.epoch)
    best_val_loss = float(min(val_losses)) if val_losses else 0.0
    
    summary_results = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED_VALIDATION_ONLY",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "training_duration_seconds": round(training_duration, 2),
        "training_duration_minutes": round(training_duration / 60.0, 2),
        "epochs_completed": len(history.epoch),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "hyperparameters": {
            "beta": 0.999,
            "weighting_strategy": "effective_number_of_samples",
            "learning_rate": lr,
            "batch_size": batch_size,
            "finetune_layers": finetune_layers,
            "seed": seed,
            "img_size": img_size
        },
        "validation_results_all": val_all_metrics,
        "validation_results_ham": val_ham_metrics,
        "history": {k: [round(float(v), 5) for v in vals] for k, vals in history.history.items()}
    }
    
    with open(OUTPUT_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
        
    print("\n================== VALIDATION EVALUATION RESULTS ==================")
    print(f"Full Validation Split (N={val_all_metrics['sample_count']}):")
    print(f"  Accuracy:         {val_all_metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:     {val_all_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:         {val_all_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {val_all_metrics['weighted_f1']:.4f}")
    print(f"  Mel Recall:       {val_all_metrics['melanoma_metrics']['recall_sensitivity']:.4f} ({val_all_metrics['melanoma_metrics']['support']} cases)")
    print(f"  Mel Precision:    {val_all_metrics['melanoma_metrics']['precision']:.4f}")
    print(f"  Mel False Alarms: {val_all_metrics['melanoma_metrics']['false_alarms_count']} / {val_all_metrics['melanoma_metrics']['non_melanoma_total']} ({val_all_metrics['melanoma_metrics']['false_alarm_rate']:.1%})")
    print(f"  Macro ROC-AUC:    {val_all_metrics['macro_roc_auc']:.4f}")
    print(f"  Macro PR-AUC:     {val_all_metrics['macro_pr_auc']:.4f}")
    print("\nHAM-Only Validation Split (N={val_ham_metrics['sample_count']}):")
    print(f"  Accuracy:         {val_ham_metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:     {val_ham_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:         {val_ham_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {val_ham_metrics['weighted_f1']:.4f}")
    print(f"  Mel Recall:       {val_ham_metrics['melanoma_metrics']['recall_sensitivity']:.4f} ({val_ham_metrics['melanoma_metrics']['support']} cases)")
    print(f"  Mel Precision:    {val_ham_metrics['melanoma_metrics']['precision']:.4f}")
    print(f"  Mel False Alarms: {val_ham_metrics['melanoma_metrics']['false_alarms_count']} / {val_ham_metrics['melanoma_metrics']['non_melanoma_total']} ({val_ham_metrics['melanoma_metrics']['false_alarm_rate']:.1%})")
    print("===================================================================\n")
    print(f"Experiment outputs saved cleanly to: {OUTPUT_DIR}")
    print("Frozen test set was NOT evaluated, preserving separation for separate approval gate.")


if __name__ == "__main__":
    run_experiment()
