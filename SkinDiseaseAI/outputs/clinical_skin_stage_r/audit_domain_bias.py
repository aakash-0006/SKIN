"""Audit domain bias and evaluate background-only predictor for Stage R."""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def analyze_background_color(image_path):
    """Sample corner pixels (top-left, top-right, bottom-left, bottom-right)."""
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            arr = np.array(im, dtype=np.float32)
            # Sample 10x10 corner patches
            c_tl = arr[:10, :10].mean(axis=(0, 1))
            c_tr = arr[:10, -10:].mean(axis=(0, 1))
            c_bl = arr[-10:, :10].mean(axis=(0, 1))
            c_br = arr[-10:, -10:].mean(axis=(0, 1))
            bg_mean = np.mean([c_tl, c_tr, c_bl, c_br], axis=0) # [R, G, B]
            bg_std = np.std([c_tl, c_tr, c_bl, c_br], axis=0).mean()
            return bg_mean, bg_std
    except Exception:
        return None, None

def main():
    print("=== Stage R: Domain Bias & Background Analysis ===")
    df = pd.read_csv("data_normal_disease/clinical_stage_q/manifest_all_local_clinical.csv")
    
    # Filter to Stage R training sources only:
    # 1. Normal Hands (irregularities = 0) -> coarse_label == 'normal_looking_skin'
    # 2. Eczema vs Tinea -> source_dataset == 'Eczema_vs_Tinea'
    mask_normal = (df["source_dataset"] == "11k_Normal_Hands") & (df["coarse_label"] == "normal_looking_skin")
    mask_disease = df["source_dataset"] == "Eczema_vs_Tinea"
    df_cohort = df[mask_normal | mask_disease].copy().reset_index(drop=True)
    
    # Label: 0 = Normal Skin, 1 = Potentially Abnormal Condition
    df_cohort["target"] = (df_cohort["source_dataset"] == "Eczema_vs_Tinea").astype(int)
    
    print(f"Cohort size: {len(df_cohort)} (Normal: {(df_cohort['target']==0).sum()}, Abnormal: {(df_cohort['target']==1).sum()})")
    print(pd.crosstab(df_cohort["source_dataset"], df_cohort["split"]))

    # Sample 500 images per class for background profiling
    print("\nExtracting background corner features from sample...")
    rng = np.random.RandomState(42)
    sample_normal = df_cohort[df_cohort["target"] == 0].sample(min(500, (df_cohort['target']==0).sum()), random_state=rng)
    sample_disease = df_cohort[df_cohort["target"] == 1].sample(min(500, (df_cohort['target']==1).sum()), random_state=rng)
    sample_df = pd.concat([sample_normal, sample_disease]).reset_index(drop=True)

    X_bg = []
    y_bg = []
    for _, row in sample_df.iterrows():
        bg_m, bg_s = analyze_background_color(row["image_path"])
        if bg_m is not None:
            # Features: [Mean_R, Mean_G, Mean_B, Corner_Variance]
            X_bg.append([bg_m[0], bg_m[1], bg_m[2], bg_s])
            y_bg.append(row["target"])

    X_bg = np.array(X_bg, dtype=np.float32)
    y_bg = np.array(y_bg, dtype=int)

    normal_bg = X_bg[y_bg == 0]
    disease_bg = X_bg[y_bg == 1]

    print("\n--- Background Color Statistics (RGB 0-255) ---")
    print(f"Normal Hands Background Mean RGB: [{normal_bg[:,0].mean():.1f}, {normal_bg[:,1].mean():.1f}, {normal_bg[:,2].mean():.1f}] (Near-pure white ~ 240-255)")
    print(f"Disease Skin Background Mean RGB: [{disease_bg[:,0].mean():.1f}, {disease_bg[:,1].mean():.1f}, {disease_bg[:,2].mean():.1f}] (Natural ambient ~ 120-180)")

    # Test Background-Only Predictor
    train_idx = np.arange(0, len(X_bg), 2)
    test_idx = np.arange(1, len(X_bg), 2)
    
    clf = LogisticRegression()
    clf.fit(X_bg[train_idx], y_bg[train_idx])
    preds = clf.predict_proba(X_bg[test_idx])[:, 1]
    bg_auc = roc_auc_score(y_bg[test_idx], preds)
    bg_acc = accuracy_score(y_bg[test_idx], (preds >= 0.5).astype(int))
    
    print("\n--- Background-Only Control Classifier Result ---")
    print(f"Background-Only ROC-AUC: {bg_auc:.4f}")
    print(f"Background-Only Accuracy: {bg_acc*100:.2f}%")
    print("\nCRITICAL INSIGHT:")
    if bg_auc > 0.90:
        print("CONFIRMED DOMAIN BIAS: A naive model looking ONLY at image corners can achieve >90% accuracy!")
        print("To learn genuine skin features, the model MUST utilize skin region extraction / hand cropping and heavy background augmentation.")
    
    # Save results to json
    import json
    res = {
        "normal_bg_mean": normal_bg.mean(axis=0).tolist(),
        "disease_bg_mean": disease_bg.mean(axis=0).tolist(),
        "background_only_auc": float(bg_auc),
        "background_only_acc": float(bg_acc)
    }
    with open("outputs/clinical_skin_stage_r/domain_bias_analysis.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
