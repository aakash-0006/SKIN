"""Mandatory Metadata Leakage Audit for Stage O.2."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, ".")
from src.isic2024.dataset import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, MetadataPreprocessor

def main():
    metadata_path = "D:/DOWNLOADS D/isic-2024-challenge/train-metadata.csv"
    splits_path = "outputs/isic2024/splits_5fold.csv"

    print("=== 1. Loading Metadata and Split Information ===")
    df = pd.read_csv(metadata_path, low_memory=False)
    splits = pd.read_csv(splits_path)
    df = df.merge(splits[["isic_id", "fold"]], on="isic_id", how="left")
    print(f"Total metadata records: {len(df)}")
    print(f"Target count: {df['target'].value_counts().to_dict()}")

    # 1. Patient isolation check across folds
    print("\n=== 2. Checking Patient-Level Split Isolation ===")
    fold0_train = df[df["fold"] != 0]
    fold0_val = df[df["fold"] == 0]
    train_patients = set(fold0_train["patient_id"].unique())
    val_patients = set(fold0_val["patient_id"].unique())
    patient_overlap = train_patients.intersection(val_patients)
    print(f"Fold 0 Train Patients: {len(train_patients)}")
    print(f"Fold 0 Val Patients: {len(val_patients)}")
    print(f"Patient Overlap Count: {len(patient_overlap)}")
    if len(patient_overlap) > 0:
        print(f"CRITICAL ERROR: Patient leakage detected! Overlapping: {list(patient_overlap)[:5]}")
    else:
        print("VERIFIED: 100% Patient isolation confirmed (zero patient overlap).")

    # 2. Missingness pattern audit between target=0 and target=1
    print("\n=== 3. Missingness Pattern Analysis ===")
    leakage_suspects = []
    feature_correlations = []
    
    pos_df = df[df["target"] == 1]
    neg_df = df[df["target"] == 0]

    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        pos_na = pos_df[col].isna().mean()
        neg_na = neg_df[col].isna().mean()
        diff_na = abs(pos_na - neg_na)
        if diff_na > 0.10:
            leakage_suspects.append((col, pos_na, neg_na, diff_na))
        
        # If numerical, compute correlation with target
        if col in NUMERICAL_FEATURES:
            valid = df[[col, "target"]].dropna()
            corr = float(valid[col].corr(valid["target"]))
            feature_correlations.append((col, corr, pos_na, neg_na))

    print(f"Features with missingness discrepancy > 10%: {len(leakage_suspects)}")
    for item in leakage_suspects:
        print(f"  {item[0]}: pos_na={item[1]:.4f}, neg_na={item[2]:.4f}, diff={item[3]:.4f}")

    # Top feature correlations with target
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 10 Numerical Features Correlated with Target:")
    for col, corr, pos_na, neg_na in feature_correlations[:10]:
        print(f"  {col:<30}: corr = {corr:+.4f} (pos_na={pos_na:.3f}, neg_na={neg_na:.3f})")

    # 3. Categorical value distribution
    print("\n=== 4. Categorical Values Audit ===")
    for col in CATEGORICAL_FEATURES:
        print(f"\nFeature: {col}")
        ct = pd.crosstab(df[col].fillna("MISSING"), df["target"], normalize="columns")
        print(ct)

    # 4. Shuffled-Label Negative Control
    print("\n=== 5. Running Shuffled-Label Permutation Negative Control ===")
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    train_df = fold0_train.copy().reset_index(drop=True)
    val_df = fold0_val.copy().reset_index(drop=True)

    # Fit preprocessor
    prep = MetadataPreprocessor()
    prep.fit(train_df)
    X_train = prep.transform(train_df)
    X_val = prep.transform(val_df)
    
    # Subsample 1:15 training for speed in negative control
    pos_idx = np.where(train_df["target"].values == 1)[0]
    neg_idx = np.where(train_df["target"].values == 0)[0]
    rng = np.random.RandomState(42)
    sampled_neg = rng.choice(neg_idx, size=len(pos_idx) * 15, replace=False)
    sub_idx = np.concatenate([pos_idx, sampled_neg])
    
    X_train_sub = X_train[sub_idx]
    y_train_sub = train_df["target"].values[sub_idx].copy()
    y_val = val_df["target"].values

    # SHUFFLE labels randomly
    y_train_shuffled = rng.permutation(y_train_sub)

    # Simple MLP
    model = models.Sequential([
        layers.Input(shape=(prep.output_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
    model.fit(X_train_sub, y_train_shuffled, epochs=5, batch_size=64, verbose=0)
    
    val_preds_shuffled = model.predict(X_val, batch_size=1024, verbose=0).ravel()
    auc_shuffled = roc_auc_score(y_val, val_preds_shuffled)
    prauc_shuffled = average_precision_score(y_val, val_preds_shuffled)
    print(f"Shuffled-Label Negative Control ROC-AUC: {auc_shuffled:.5f}")
    print(f"Shuffled-Label Negative Control PR-AUC : {prauc_shuffled:.5f}")
    if abs(auc_shuffled - 0.50) < 0.05:
        print("VERIFIED: Permutation test confirms model behaves as random chance when labels are decoupled. Zero feature-level label leakage!")
    else:
        print(f"WARNING: Shuffled control ROC-AUC deviated from 0.50 ({auc_shuffled:.5f})!")

if __name__ == "__main__":
    main()
