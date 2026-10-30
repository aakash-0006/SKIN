"""Controlled training pipeline for Stage R Clinical Skin Appearance Model."""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Set seeds
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage R Clinical Skin Appearance Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 200 samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


class ClinicalDatasetSequence(tf.keras.utils.Sequence):
    """Data sequence loading clinical images with domain-invariant augmentation."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        img_size: int = 224,
        is_training: bool = True,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = (img_size, img_size)
        self.is_training = is_training
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(self.df))
        if self.is_training:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.is_training:
            self.rng.shuffle(self.indices)

    def _augment(self, im: Image.Image) -> Image.Image:
        """Apply random crop & color jitter to suppress background contrast cues."""
        w, h = im.size
        # 1. Random crop (focusing toward skin center)
        if self.rng.rand() > 0.3:
            crop_scale = self.rng.uniform(0.70, 0.95)
            new_w, new_h = int(w * crop_scale), int(h * crop_scale)
            left = self.rng.randint(0, max(1, w - new_w))
            top = self.rng.randint(0, max(1, h - new_h))
            im = im.crop((left, top, left + new_w, top + new_h))

        # 2. Random horizontal flip
        if self.rng.rand() > 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        return im

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        b_size = len(batch_idx)

        x_batch = np.zeros((b_size, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        y_batch = np.zeros((b_size, 1), dtype=np.float32)

        for i, b_i in enumerate(batch_idx):
            row = self.df.iloc[b_i]
            if "target" in row:
                y_batch[i, 0] = float(row["target"])
            try:
                with Image.open(row["image_path"]) as im:
                    im = im.convert("RGB")
                    if self.is_training:
                        im = self._augment(im)
                    im = im.resize(self.img_size, Image.Resampling.BILINEAR)
                    # MobileNetV3 expects raw float32 in [0, 255]
                    x_batch[i] = np.array(im, dtype=np.float32)
            except Exception:
                # Fallback zero array if unreadable
                pass

        return x_batch, y_batch


def build_clinical_model(input_shape=(224, 224, 3), learning_rate: float = 1e-4) -> tf.keras.Model:
    """Build MobileNetV3Small classification model for clinical skin appearance."""
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = True

    inputs = layers.Input(shape=input_shape, name="clinical_image_input")
    x = base_model(inputs)
    x = layers.Dense(128, activation="relu", name="clinical_dense_1")(x)
    x = layers.Dropout(0.3, name="clinical_dropout_1")(x)
    x = layers.Dense(32, activation="relu", name="clinical_dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="abnormal_appearance_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="clinical_skin_appearance_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


def evaluate_clinical_predictions(y_true, y_probs, threshold=0.5):
    """Compute comprehensive diagnostic metrics."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs).astype(float).ravel()
    y_pred_bin = (y_probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))

    return {
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "sensitivity": round(sens, 5),
        "specificity": round(spec, 5),
        "precision": round(prec, 5),
        "f1_score": round(f1, 5),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sample_counts": {"total": len(y_true), "normal": int((y_true==0).sum()), "abnormal": int((y_true==1).sum())},
    }


def main():
    args = parse_args()
    set_seeds(args.seed)

    print("=" * 80)
    print("STAGE R: CLINICAL SKIN APPEARANCE MODEL TRAINING")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Smoke Test: {args.smoke_test}")
    print("=" * 80)

    # 1. Load manifest and filter strictly per specification
    manifest_path = "data_normal_disease/clinical_stage_q/manifest_all_local_clinical.csv"
    df = pd.read_csv(manifest_path)

    # Select: Normal Hands (irregularities = 0) and Eczema vs Tinea
    # Explicitly excludes the 164 hand irregularity images and Stanford DDI
    mask_normal = (df["source_dataset"] == "11k_Normal_Hands") & (df["coarse_label"] == "normal_looking_skin")
    mask_disease = df["source_dataset"] == "Eczema_vs_Tinea"
    df_cohort = df[mask_normal | mask_disease].copy().reset_index(drop=True)

    # 0 = Normal Hand Appearance, 1 = Potentially Abnormal Clinical Condition
    df_cohort["target"] = (df_cohort["source_dataset"] == "Eczema_vs_Tinea").astype(int)

    train_df = df_cohort[df_cohort["split"] == "train"].copy().reset_index(drop=True)
    val_df = df_cohort[df_cohort["split"] == "val"].copy().reset_index(drop=True)
    test_df = df_cohort[df_cohort["split"] == "test"].copy().reset_index(drop=True)

    if args.smoke_test:
        print("\n--- Running Smoke Test Subset (100 train, 100 val) ---")
        train_df = pd.concat([
            train_df[train_df["target"] == 0].sample(50, random_state=42),
            train_df[train_df["target"] == 1].sample(50, random_state=42),
        ]).reset_index(drop=True)
        val_df = pd.concat([
            val_df[val_df["target"] == 0].sample(50, random_state=42),
            val_df[val_df["target"] == 1].sample(50, random_state=42),
        ]).reset_index(drop=True)

    print(f"Training samples: {len(train_df)} (Normal: {(train_df['target']==0).sum()}, Abnormal: {(train_df['target']==1).sum()})")
    print(f"Validation samples: {len(val_df)} (Normal: {(val_df['target']==0).sum()}, Abnormal: {(val_df['target']==1).sum()})")

    # Balanced class weights for training
    n_normal = (train_df["target"] == 0).sum()
    n_abnormal = (train_df["target"] == 1).sum()
    total = len(train_df)
    class_weights = {
        0: float(total / (2.0 * n_normal)),
        1: float(total / (2.0 * n_abnormal)),
    }
    print(f"Class weights: Normal={class_weights[0]:.3f}, Abnormal={class_weights[1]:.3f}")

    # 2. Sequences
    train_seq = ClinicalDatasetSequence(train_df, batch_size=args.batch_size, img_size=args.img_size, is_training=True)
    val_seq = ClinicalDatasetSequence(val_df, batch_size=args.batch_size * 2, img_size=args.img_size, is_training=False)

    # 3. Model
    model = build_clinical_model(input_shape=(args.img_size, args.img_size, 3), learning_rate=args.lr)

    ckpt_path = "models/clinical_skin_stage_r/clinical_skin_stage_r_best.keras"
    out_dir = Path("outputs/clinical_skin_stage_r")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_roc_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
    ]

    print("\nBeginning training...")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # 4. Evaluation on Validation Set
    print("\nEvaluating on full validation set...")
    val_probs = model.predict(val_seq, verbose=1).ravel()
    val_targets = val_df["target"].values
    val_metrics = evaluate_clinical_predictions(val_targets, val_probs, threshold=0.5)

    print("\n--- Validation Performance ---")
    print(json.dumps(val_metrics, indent=2))
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    np.save(out_dir / "val_probabilities.npy", val_probs)

    # 5. Independent Domain Generalization Evaluation on Stanford DDI
    print("\nEvaluating on Independent External Stanford DDI (656 images)...")
    ddi_df = df[df["source_dataset"] == "Stanford_DDI"].copy().reset_index(drop=True)
    ddi_seq = ClinicalDatasetSequence(ddi_df, batch_size=args.batch_size * 2, img_size=args.img_size, is_training=False)
    ddi_probs = model.predict(ddi_seq, verbose=1).ravel()

    # In DDI, all 656 images represent visible conditions (485 benign lesions, 171 malignant neoplasms)
    # A well-generalized clinical appearance model should predict high abnormal probability (>0.50)
    ddi_pred_abnormal_rate = float((ddi_probs >= 0.50).mean())
    ddi_mean_prob = float(ddi_probs.mean())
    ddi_benign_mean = float(ddi_probs[ddi_df["is_malignant"] == False].mean())
    ddi_malig_mean = float(ddi_probs[ddi_df["is_malignant"] == True].mean())

    ddi_eval = {
        "total_images": len(ddi_df),
        "predicted_abnormal_rate_at_t05": round(ddi_pred_abnormal_rate, 4),
        "mean_abnormal_probability": round(ddi_mean_prob, 4),
        "benign_lesions_mean_prob": round(ddi_benign_mean, 4),
        "malignant_lesions_mean_prob": round(ddi_malig_mean, 4),
        "skin_tone_breakdown": {
            "FST_12": round(float(ddi_probs[ddi_df["skin_tone"] == "12"].mean()), 4),
            "FST_34": round(float(ddi_probs[ddi_df["skin_tone"] == "34"].mean()), 4),
            "FST_56": round(float(ddi_probs[ddi_df["skin_tone"] == "56"].mean()), 4),
        }
    }
    print("\n--- Stanford DDI Domain Generalization Evaluation ---")
    print(json.dumps(ddi_eval, indent=2))
    with open(out_dir / "ddi_generalization_eval.json", "w") as f:
        json.dump(ddi_eval, f, indent=2)

    # 6. Calibrate Uncertainty Rule
    # Rule: If prediction is between [0.35, 0.65] (entropy is high), classify as "Uncertain"
    uncertain_mask = (val_probs >= 0.35) & (val_probs <= 0.65)
    uncertain_rate = float(uncertain_mask.mean())
    print(f"\nUncertainty Gate ([0.35, 0.65] interval): {uncertain_rate*100:.2f}% of validation samples flagged as uncertain.")

    print(f"\nTraining complete. Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
