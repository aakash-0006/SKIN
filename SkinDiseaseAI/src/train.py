"""Fine-tune the skin lesion classifier with online image augmentation and balanced class weighting.

Protects original models (best_model.keras, isic_finetuned.keras) and frozen splits (splits.csv, combined_splits.csv).
Saves new model to models/isic_augmented_v1.keras with distinct history and report files.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import CLASSES, PROJECT_DIR, load_config, resolve_split_paths
from src.isic import HAM_TO_IDX, inspect_isic, load_isic_training_frame, _resolve
from src.preprocessing import make_dataset, build_augmentation_layer, apply_augmentation_to_model

PROTECTED_PATHS = {
    "models/best_model.keras",
    "models/isic_finetuned.keras",
    "outputs/splits.csv",
    "outputs/combined_splits.csv",
}


def _check_output_protection(output_path):
    target = _resolve(output_path).resolve()
    for protected in PROTECTED_PATHS:
        if target == _resolve(protected).resolve():
            raise ValueError(f"PROTECTION VIOLATION: Refusing to overwrite protected file: {target}")


def _require_split_integrity(df):
    sets = {name: set(part.lesion_id.astype(str)) for name, part in df.groupby("split")}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise RuntimeError(f"Split has {len(overlap)} lesion IDs in both {left} and {right}.")
    test = df[df.split == "test"]
    if test.empty:
        raise RuntimeError("Existing split manifest has no test rows; refusing to train without preserved test data.")


def _isic_train_val(isic, seed, train_share, duplicate_ids):
    isic = isic[~isic.image.astype(str).isin(duplicate_ids)].copy()
    if isic.empty:
        raise RuntimeError("No compatible ISIC images remain after label and duplicate exclusions.")
    groups = isic.lesion_group.astype(str).values
    if len(set(groups)) == 1:
        raise RuntimeError("ISIC data has only one lesion group; cannot make grouped train/validation splits.")
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_share, random_state=seed)
    tr, va = next(splitter.split(isic, groups=groups))
    isic.loc[:, "split"] = ""
    isic.iloc[tr, isic.columns.get_loc("split")] = "train"
    isic.iloc[va, isic.columns.get_loc("split")] = "val"
    return isic


def compute_safe_class_weights(
    y_train,
    classes=CLASSES,
    strategy="balanced_capped",
    max_cap=5.0,
    smoothing_power=0.6,
):
    """Compute balanced class weights from training labels ONLY, with rare-class safeguards.

    In the training data, nv has 8,229 samples while df has 173 and vasc has 174 (~48:1 ratio).
    Standard inverse-frequency balanced weights yield ~13.6 for df/vasc and ~0.287 for nv.
    An unconstrained ~48x loss weight on rare classes produces gradient instability, noisy updates,
    and degraded precision on major classes like mel, bcc, and akiec.

    Rationale for max_cap=5.0:
    Limits the gradient multiplier ratio of rare classes to ~17:1 instead of ~48:1, providing
    meaningful loss penalties for rare-class misses without overwhelming the optimizer or distorting
    decision boundaries for critical classes such as melanoma (mel).
    """
    n_classes = len(classes)
    y_arr = np.asarray(y_train, dtype=int)
    counts = np.bincount(y_arr, minlength=n_classes)
    total = len(y_arr)

    # Standard balanced: N / (C * N_c)
    weights = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        weights[c] = total / (n_classes * counts[c]) if counts[c] > 0 else 1.0

    if strategy == "balanced_capped":
        applied_weights = np.clip(weights, a_min=None, a_max=max_cap) if max_cap else weights
    elif strategy == "power_smoothed":
        applied_weights = weights ** smoothing_power
        if max_cap is not None:
            applied_weights = np.clip(applied_weights, a_min=None, a_max=max_cap)
    elif strategy == "balanced":
        applied_weights = weights
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")

    class_weight = {int(i): float(applied_weights[i]) for i in range(n_classes)}

    nv_idx = classes.index("nv") if "nv" in classes else 0
    nv_w = applied_weights[nv_idx]

    print("\n=================== CLASS WEIGHTS & DISTRIBUTION ===================")
    print(f"Total training samples: {total} (Validation & Test labels strictly excluded)")
    print(f"Strategy: {strategy} | Max Cap: {max_cap}")
    print(f"{'Idx':<4} {'Class':<8} {'Train Count':<13} {'Raw Weight':<13} {'Applied Weight':<16} {'Ratio vs nv':<12}")
    print("-" * 68)
    for i, name in enumerate(classes):
        ratio = applied_weights[i] / nv_w if nv_w > 0 else 0
        cap_note = " [CAPPED]" if applied_weights[i] < weights[i] else ""
        print(f"{i:<4} {name:<8} {counts[i]:<13} {weights[i]:<13.4f} {applied_weights[i]:<16.4f}{cap_note} {ratio:<12.2f}x")
    print("====================================================================\n")

    log_info = {
        "strategy": strategy,
        "max_cap": max_cap,
        "total_training_samples": int(total),
        "class_counts": {name: int(counts[i]) for i, name in enumerate(classes)},
        "raw_balanced_weights": {name: round(float(weights[i]), 4) for i, name in enumerate(classes)},
        "applied_weights": {name: round(float(applied_weights[i]), 4) for i, name in enumerate(classes)},
        "ratios_vs_nv": {name: round(float(applied_weights[i] / nv_w), 2) for i, name in enumerate(classes)},
    }
    return class_weight, log_info


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Fine-tune skin lesion classifier with augmentation and class weighting.")
    ap.add_argument("--ham-root", "--root", dest="ham_root", default=cfg["dataset_root"])
    ap.add_argument("--isic-root", default=cfg.get("isic_dataset_root") or None)
    ap.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 32))
    ap.add_argument("--epochs", type=int, default=cfg.get("finetune_epochs", 10))
    ap.add_argument("--learning-rate", type=float, default=cfg.get("finetune_lr", 1e-5))
    ap.add_argument("--finetune-layers", type=int, default=cfg.get("finetune_layers", 40))
    ap.add_argument("--init-model", default="models/isic_finetuned.keras",
                    help="Starting model checkpoint (defaults to fine-tuned ISIC model)")
    ap.add_argument("--output-model", default="models/isic_augmented_v1.keras",
                    help="Path to save the new trained model")
    ap.add_argument("--combined-splits", default="outputs/combined_splits.csv",
                    help="Path to existing or target combined splits manifest")
    ap.add_argument("--augment", action="store_true", default=True,
                    help="Enable conservative online data augmentation (default: True)")
    ap.add_argument("--no-augment", action="store_false", dest="augment",
                    help="Disable online data augmentation")
    ap.add_argument("--weight-strategy", default="balanced_capped",
                    choices=["balanced_capped", "power_smoothed", "balanced"],
                    help="Class weighting strategy")
    ap.add_argument("--weight-cap", type=float, default=cfg.get("class_weight_cap", 5.0),
                    help="Maximum weight cap for rare classes (default: 5.0)")
    ap.add_argument("--history-csv", default="outputs/augmentation_training_history.csv",
                    help="CSV file path to record training epoch history")
    ap.add_argument("--report-json", default="outputs/augmentation_training_report.json",
                    help="JSON file path for detailed training summary report")
    ap.add_argument("--config-json", default="outputs/augmentation_config.json",
                    help="JSON file path for training run configuration")
    ap.add_argument("--resume", action="store_true",
                    help="Continue training from output-model if it already exists")
    ap.add_argument("--inspection-acknowledged", action="store_true",
                    help="Confirm ISIC inspection report has been reviewed")
    args = ap.parse_args()

    # Safety checks
    _check_output_protection(args.output_model)
    _check_output_protection(args.history_csv)
    _check_output_protection(args.report_json)

    if args.batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0 or args.finetune_layers < 1:
        ap.error("batch size, epochs and fine-tuned layers must be positive; learning rate must be greater than zero")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    tf.keras.utils.set_random_seed(cfg["seed"])
    out = PROJECT_DIR / "outputs"
    models_dir = PROJECT_DIR / "models"
    out.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    splits_manifest_path = _resolve(args.combined_splits)
    ham_splits_path = _resolve(cfg.get("existing_splits_path", "outputs/splits.csv"))
    baseline_ham = resolve_split_paths(pd.read_csv(ham_splits_path), args.ham_root)
    _require_split_integrity(baseline_ham)

    # 1. Load or build combined splits manifest
    if splits_manifest_path.is_file():
        print(f"Loading existing verified split manifest: {splits_manifest_path}")
        combined = pd.read_csv(splits_manifest_path)
        _require_split_integrity(combined)
        # Verify frozen test partition remains 100% byte-for-byte identical to baseline HAM test split
        old_test = baseline_ham.loc[baseline_ham.split == "test", "image_id"].astype(str).tolist()
        new_test = combined.loc[combined.split == "test", "image"].astype(str).tolist()
        if old_test != new_test:
            raise RuntimeError("Existing combined manifest test split does not match baseline HAM test split!")
    else:
        # Build manifest if not present (requires isic_root)
        if not args.isic_root:
            ap.error("Combined splits file not found and isic_root is not provided.")
        report_path = _resolve(cfg.get("inspection_report_path", "outputs/isic_dataset_report.json"))
        if not report_path.is_file():
            ap.error(f"Inspection report is missing: {report_path}")
        if not args.inspection_acknowledged:
            ap.error("Review the inspection report first, then rerun with --inspection-acknowledged.")
        reviewed_report = json.loads(report_path.read_text(encoding="utf-8"))
        isic_rows, ham_scan, report = inspect_isic(
            args.isic_root, args.ham_root, ham_splits_path,
            report_path, cfg.get("near_duplicate_hamming_threshold", 4)
        )
        if report["scan_signature"] != reviewed_report["scan_signature"]:
            raise RuntimeError("Dataset integrity changed since reviewed report.")

        duplicate_ids = set()
        duplicate_ids.update(map(str, report["shared_image_ids_with_ham"]))
        duplicate_ids.update(str(row["image_id"]) for row in report["duplicate_image_filenames"])
        for pair in report["exact_duplicate_pairs_with_ham"] + report["near_duplicate_pairs_with_ham"]:
            duplicate_ids.add(str(pair["isic_image"]))
        for pair in report["exact_duplicate_pairs_within_isic"] + report["near_duplicate_pairs_within_isic"]:
            duplicate_ids.update((str(pair["image_a"]), str(pair["image_b"])))

        ham_excluded_ids = {str(p["image_id"]) for p in ham_scan if p.get("error") and p.get("split") != "test"}
        ham_conflicts = (report["exact_duplicate_pairs_within_ham_across_splits"] +
                         report["near_duplicate_pairs_within_ham_across_splits"])
        for pair in ham_conflicts:
            a, b = pair["image_a"], pair["image_b"]
            sa, sb = pair["split_a"], pair["split_b"]
            if sa == "test":
                ham_excluded_ids.add(b)
            elif sb == "test":
                ham_excluded_ids.add(a)
            elif sa == "train" or sb == "train":
                ham_excluded_ids.add(a if sa == "train" else b)
            else:
                ham_excluded_ids.add(a)

        isic = load_isic_training_frame(isic_rows)
        ham = baseline_ham[baseline_ham.split.isin(["train", "val"])].copy()
        ham = ham[~ham.image_id.astype(str).isin(ham_excluded_ids)].copy()
        ham["y"] = ham.dx.map(HAM_TO_IDX).astype("int32")
        ham["image"] = ham.image_id.astype(str)
        ham["dataset"] = "HAM10000"
        isic["dataset"] = "ISIC2019"
        isic_share = cfg["train_frac"] / (cfg["train_frac"] + cfg["val_frac"])
        isic = _isic_train_val(isic, cfg["seed"], isic_share, duplicate_ids)

        combined = pd.concat([
            ham[["image", "path", "dx", "y", "lesion_id", "split", "dataset"]],
            isic.assign(lesion_id=isic.lesion_group)[["image", "path", "dx", "y", "lesion_id", "split", "dataset"]],
            baseline_ham[baseline_ham.split == "test"].assign(dataset="HAM10000", image=lambda d: d.image_id.astype(str))
                .assign(y=lambda d: d.dx.map(HAM_TO_IDX))[["image", "path", "dx", "y", "lesion_id", "split", "dataset"]],
        ], ignore_index=True)
        combined["y"] = combined.y.astype("int32")
        _require_split_integrity(combined)
        combined["source_image_id"] = combined.image
        combined.to_csv(splits_manifest_path, index=False)
        print(f"Created new combined split manifest at: {splits_manifest_path}")

    # 2. Partition dataset
    train = combined[combined.split == "train"].copy()
    val = combined[combined.split == "val"].copy()
    test = combined[combined.split == "test"].copy()

    if train.empty or val.empty or test.empty:
        raise RuntimeError(f"Split partition empty: train={len(train)}, val={len(val)}, test={len(test)}")

    for part, name in ((train, "train"), (val, "val"), (test, "test")):
        missing = [p for p in part.path if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing from {name} split; sample: {missing[:3]}")

    print(f"Partitions verified: Train={len(train)}, Val={len(val)}, Preserved Test={len(test)}")

    # 3. Calculate safe class weights ONLY from train partition
    class_weight, weight_info = compute_safe_class_weights(
        train.y.values,
        classes=CLASSES,
        strategy=args.weight_strategy,
        max_cap=args.weight_cap,
    )

    # 4. Create datasets
    train_ds = make_dataset(train.path, train.y, cfg["img_size"], args.batch_size, training=True, seed=cfg["seed"])
    val_ds = make_dataset(val.path, val.y, cfg["img_size"], args.batch_size)

    # 5. Load starting model and configure augmentation
    init_path = _resolve(args.init_model)
    if not init_path.is_file():
        # Fallback to model_path in config if init_model doesn't exist
        fallback_path = _resolve(cfg.get("model_path", "models/best_model.keras"))
        if fallback_path.is_file():
            print(f"Note: {init_path} not found. Falling back to base model: {fallback_path}")
            init_path = fallback_path
        else:
            raise FileNotFoundError(f"Starting model checkpoint not found: {init_path}")

    output_path = _resolve(args.output_model)
    print(f"Loading initial model from: {init_path}")
    source_model = tf.keras.models.load_model(init_path, compile=False)

    if args.augment:
        print("Configuring conservative online image augmentation layer:")
        print("  - Random horizontal flip (no vertical flip)")
        print("  - Random rotation: ±20 degrees")
        print("  - Random zoom: ±8%")
        print("  - Random translation: ±5% (horizontal & vertical)")
        print("  - Dimensions: 224x224, Scale: [0, 255]")
        print("  - Active ONLY during training; exact identity during eval")
        model = apply_augmentation_to_model(source_model)
    else:
        print("Augmentation disabled by request.")
        model = source_model

    if tuple(model.input_shape[1:]) != (cfg["img_size"], cfg["img_size"], 3) or model.output_shape[-1] != len(CLASSES):
        raise ValueError(f"Model shape mismatch: input={model.input_shape}, output={model.output_shape}")

    # Fine-tune specified number of top layers of EfficientNetB0
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    frozen_prefix = max(0, len(base.layers) - args.finetune_layers)
    for layer in base.layers[:frozen_prefix]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_csv = _resolve(args.history_csv)
    report_json = _resolve(args.report_json)
    config_json = _resolve(args.config_json)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(str(output_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(history_csv), append=args.resume),
    ]

    print(f"\nStarting fine-tuning for {args.epochs} epochs (Batch size={args.batch_size}, LR={args.learning_rate})...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    training_duration = time.time() - start_time

    # Best epoch stats
    val_losses = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else len(history.epoch)
    best_val_loss = float(min(val_losses)) if val_losses else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "source_checkpoint": str(init_path),
        "saved_model_path": str(output_path),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_duration_seconds": round(training_duration, 2),
        "training_duration_minutes": round(training_duration / 60.0, 2),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "finetuned_layers": args.finetune_layers,
            "img_size": cfg["img_size"],
            "seed": cfg["seed"],
        },
        "augmentation": {
            "enabled": args.augment,
            "type": "online_keras_sequential",
            "horizontal_flip": True,
            "vertical_flip": False,
            "max_rotation_degrees": 20.0,
            "zoom_range": "±8%",
            "translation_range": "±5%",
            "active_in_training": True,
            "active_in_validation_testing": False,
        },
        "class_weighting": weight_info,
        "dataset_splits": {
            "manifest": str(splits_manifest_path),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "train_counts_by_class": {c: int((train.dx == c).sum()) for c in CLASSES},
            "val_counts_by_class": {c: int((val.dx == c).sum()) for c in CLASSES},
            "test_counts_by_class": {c: int((test.dx == c).sum()) for c in CLASSES},
        },
        "history": {k: [round(float(x), 5) for x in v] for k, v in history.history.items()},
    }

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved training report to: {report_json}")

    run_config = {
        **cfg,
        "source_model": str(init_path),
        "model_path": str(output_path),
        "class_to_idx": {c: i for i, c in enumerate(CLASSES)},
        "epochs_run": len(history.epoch),
        "best_val_loss": best_val_loss,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "fine_tuned_layers": args.finetune_layers,
        "augmentation_enabled": args.augment,
        "class_weight_strategy": args.weight_strategy,
        "class_weight_cap": args.weight_cap,
    }
    config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Saved run config to: {config_json}")
    print(f"Model saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
