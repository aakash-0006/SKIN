"""Training pipeline for ISIC 2024 dedicated binary malignancy models."""
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.isic2024.dataset import (
    MetadataPreprocessor,
    ISIC2024Sequence,
)
from src.isic2024.models import (
    build_multimodal_model,
    build_image_only_model,
    build_meta_only_model,
    BinaryFocalLoss,
)
from src.isic2024.metrics import evaluate_binary_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train ISIC 2024 Binary Malignancy Model (Stage O.2)")
    parser.add_argument("--data-dir", type=str, default=r"D:\DOWNLOADS D\isic-2024-challenge", help="ISIC 2024 dataset root")
    parser.add_argument("--splits-csv", type=str, default="outputs/isic2024/splits_5fold.csv", help="5-fold patient split CSV")
    parser.add_argument("--output-dir", type=str, default="outputs/isic2024/stage_o2", help="Directory for training outputs and metrics")
    parser.add_argument("--models-dir", type=str, default="models/isic2024_stage_o2", help="Directory for saved model checkpoints")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index (0 to 4)")
    parser.add_argument("--mode", type=str, choices=["fusion", "image_only", "meta_only"], default="fusion", help="Model variant")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Phase 1 / Tabular")
    parser.add_argument("--neg-ratio", type=int, default=15, help="Negative-to-positive sampling ratio per epoch")
    parser.add_argument("--use-focal", action="store_true", help="Use focal loss instead of Binary Cross-Entropy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_h5 = data_dir / "train-image.hdf5"
    train_meta_csv = data_dir / "train-metadata.csv"
    splits_csv = Path(args.splits_csv)
    output_dir = Path(args.output_dir) / f"fold_{args.fold}_{args.mode}"
    models_dir = Path(args.models_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"ISIC 2024 BINARY MALIGNANCY TRAINING — STAGE O.2 (Fold: {args.fold}, Mode: {args.mode})")
    print(f"Loss: {'FocalLoss' if args.use_focal else 'BinaryCrossentropy'} | Neg-Ratio: 1:{args.neg_ratio}")
    print("=" * 80)

    # 1. Load splits and full metadata
    print("Loading splits and metadata...")
    df_splits = pd.read_csv(splits_csv)
    df_all_meta = pd.read_csv(train_meta_csv, low_memory=False)

    df_merged = pd.merge(df_all_meta, df_splits[["isic_id", "fold"]], on="isic_id")
    train_df = df_merged[df_merged["fold"] != args.fold].copy().reset_index(drop=True)
    val_df = df_merged[df_merged["fold"] == args.fold].copy().reset_index(drop=True)

    print(f"Dataset split loaded: Train={len(train_df):,} (Pos: {train_df['target'].sum():,}), Val={len(val_df):,} (Pos: {val_df['target'].sum():,})")

    # 2. Metadata preprocessing
    print("Fitting MetadataPreprocessor on training partition...")
    preprocessor = MetadataPreprocessor()
    preprocessor.fit(train_df)
    meta_train = preprocessor.transform(train_df)
    meta_val = preprocessor.transform(val_df)

    # 3. Dynamic negative resampling training sequence (resamples fresh negatives each epoch)
    train_seq = ISIC2024Sequence(
        train_df,
        str(train_h5),
        meta_train,
        batch_size=args.batch_size,
        mode=args.mode,
        shuffle=True,
        resample_negatives=True,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    # Fast deterministic validation monitor for intermediate epoch callbacks
    pos_val_idx = np.where(val_df["target"].values == 1)[0]
    neg_val_idx = np.where(val_df["target"].values == 0)[0]
    rng_val = np.random.RandomState(args.seed)
    sampled_val_neg = rng_val.choice(neg_val_idx, size=min(len(neg_val_idx), len(pos_val_idx) * 20), replace=False)
    val_monitor_idx = np.concatenate([pos_val_idx, sampled_val_neg])
    val_monitor_df = val_df.iloc[val_monitor_idx].reset_index(drop=True)
    meta_val_monitor = meta_val[val_monitor_idx]

    val_monitor_seq = ISIC2024Sequence(
        val_monitor_df,
        str(train_h5),
        meta_val_monitor,
        batch_size=args.batch_size * 2,
        mode=args.mode,
        shuffle=False,
        resample_negatives=False,
    )

    # Complete natural prevalence validation cohort for final reporting
    full_val_seq = ISIC2024Sequence(
        val_df,
        str(train_h5),
        meta_val,
        batch_size=args.batch_size * 2,
        mode=args.mode,
        shuffle=False,
        resample_negatives=False,
    )

    ckpt_path = models_dir / f"isic2024_{args.mode}_fold{args.fold}_best.keras"

    # 4. Model construction and training execution
    if args.mode == "fusion":
        print("\n--- PHASE 1: Warmup with Frozen Backbone (5 Epochs) ---")
        model = build_multimodal_model(
            image_shape=(128, 128, 3),
            meta_dim=preprocessor.output_dim,
            learning_rate=args.lr,
            use_focal_loss=args.use_focal,
            backbone_trainable=False,
        )

        phase1_callbacks = [
            tf.keras.callbacks.CSVLogger(str(output_dir / "phase1_history.csv")),
            tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_pr_auc", mode="max", save_best_only=True),
        ]

        model.fit(
            train_seq,
            validation_data=val_monitor_seq,
            epochs=5,
            callbacks=phase1_callbacks,
        )

        print("\n--- PHASE 2: Unfreeze Upper Backbone & Fine-Tune (lr=1e-5, up to 10 Epochs) ---")
        from src.isic2024.models import unfreeze_upper_backbone
        model = unfreeze_upper_backbone(model, unfreeze_layers=30, learning_rate=1e-5)

        phase2_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_pr_auc", mode="max", save_best_only=True),
            tf.keras.callbacks.CSVLogger(str(output_dir / "phase2_history.csv")),
        ]

        model.fit(
            train_seq,
            validation_data=val_monitor_seq,
            epochs=10,
            callbacks=phase2_callbacks,
        )

    elif args.mode == "image_only":
        print("\n--- Image-Only Training (End-to-End MobileNetV3) ---")
        model = build_image_only_model(
            image_shape=(128, 128, 3),
            learning_rate=1e-4,
            use_focal_loss=args.use_focal,
            backbone_trainable=True,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_pr_auc", mode="max", save_best_only=True),
            tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
        ]
        model.fit(
            train_seq,
            validation_data=val_monitor_seq,
            epochs=args.epochs,
            callbacks=callbacks,
        )

    else:  # meta_only
        print("\n--- Metadata-Only Tabular Training ---")
        model = build_meta_only_model(
            meta_dim=preprocessor.output_dim,
            learning_rate=args.lr,
            use_focal_loss=args.use_focal,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_pr_auc", mode="max", save_best_only=True),
            tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
        ]
        model.fit(
            train_seq,
            validation_data=full_val_seq,
            epochs=args.epochs,
            callbacks=callbacks,
        )

    # 5. Final evaluation on complete natural prevalence validation cohort
    print("\nRunning final validation evaluation on full natural cohort...")
    val_probs = model.predict(full_val_seq, verbose=1).ravel()
    val_targets = val_df["target"].values

    metrics = evaluate_binary_predictions(val_targets, val_probs, threshold=0.5, max_fpr=0.20)
    print("\nFinal Validation Results:")
    print(json.dumps(metrics, indent=2))

    with open(output_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(output_dir / "val_probabilities.npy", val_probs)
    print(f"\nTraining complete. Checkpoint: {ckpt_path} | Metrics: {output_dir / 'val_metrics.json'}")


if __name__ == "__main__":
    main()
