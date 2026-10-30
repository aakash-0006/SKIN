"""Dataset loaders, HDF5 image decoders, and metadata preprocessors for ISIC 2024."""
import io
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


# Non-leaky clinical and 3D TBP candidate features
NUMERICAL_FEATURES = [
    "age_approx",
    "clin_size_long_diam_mm",
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
    "tbp_lv_dnn_lesion_confidence",
]

CATEGORICAL_FEATURES = [
    "sex",
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location_simple",
]

# Ground-truth biopsy and diagnostic columns that MUST NEVER be fed into the model
EXCLUDED_LEAKY_COLUMNS = [
    "target",
    "iddx_full",
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "mel_mitotic_index",
    "mel_thick_mm",
]

EXCLUDED_ID_COLUMNS = [
    "isic_id",
    "patient_id",
    "lesion_id",
    "attribution",
    "copyright_license",
    "image_type",
]


class MetadataPreprocessor:
    """Preprocesses tabular metadata: median imputation, standard scaling, and one-hot encoding."""

    def __init__(self):
        self.num_medians: Dict[str, float] = {}
        self.num_means: Dict[str, float] = {}
        self.num_stds: Dict[str, float] = {}
        self.cat_categories: Dict[str, List[str]] = {}
        self.is_fitted: bool = False
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        """Compute normalization and categorical levels strictly from training partition."""
        self.num_medians = {}
        self.num_means = {}
        self.num_stds = {}
        self.feature_names = []

        # 1. Numerical statistics
        for col in NUMERICAL_FEATURES:
            median_val = float(df[col].median(skipna=True))
            filled = df[col].fillna(median_val)
            mean_val = float(filled.mean())
            std_val = float(filled.std())
            if std_val < 1e-6:
                std_val = 1.0
            self.num_medians[col] = median_val
            self.num_means[col] = mean_val
            self.num_stds[col] = std_val
            self.feature_names.append(col)

        # 2. Categorical statistics
        self.cat_categories = {}
        for col in CATEGORICAL_FEATURES:
            cats = sorted([str(x) for x in df[col].dropna().unique().tolist()])
            self.cat_categories[col] = cats
            for c in cats:
                self.feature_names.append(f"{col}_{c}")
            self.feature_names.append(f"{col}_missing")

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into a dense 2D float32 feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("MetadataPreprocessor must be fitted before transform!")

        n_samples = len(df)
        total_features = len(self.feature_names)
        output = np.zeros((n_samples, total_features), dtype=np.float32)

        feat_idx = 0
        # 1. Numericals
        for col in NUMERICAL_FEATURES:
            vals = df[col].fillna(self.num_medians[col]).values.astype(np.float32)
            normalized = (vals - self.num_means[col]) / self.num_stds[col]
            # Clip extreme numerical outliers
            output[:, feat_idx] = np.clip(normalized, -10.0, 10.0)
            feat_idx += 1

        # 2. Categoricals (one-hot encoded)
        for col in CATEGORICAL_FEATURES:
            raw_vals = df[col].astype(str).values
            for cat in self.cat_categories[col]:
                output[:, feat_idx] = (raw_vals == cat).astype(np.float32)
                feat_idx += 1
            output[:, feat_idx] = (df[col].isna()).astype(np.float32)
            feat_idx += 1

        return output

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)


class HDF5ImageDecoder:
    """Safely decodes and resizes compressed JPEG image bytes from HDF5 archive."""

    def __init__(self, h5_path: str, target_size: Tuple[int, int] = (128, 128)):
        self.h5_path = str(h5_path)
        self.target_size = target_size
        self._h5f = None

    def _get_h5(self):
        if self._h5f is None:
            self._h5f = h5py.File(self.h5_path, "r")
        return self._h5f

    def get_image(self, isic_id: str) -> np.ndarray:
        """Decode JPEG bytes from HDF5, resize, and return float32 RGB array in [0.0, 255.0] range.

        MobileNetV3 incorporates an internal Rescaling(scale=1/127.5, offset=-1.0) layer
        which maps [0.0, 255.0] to [-1.0, 1.0]. Returning raw pixel values prevents
        double normalization.
        """
        h5f = self._get_h5()
        raw_bytes = h5f[isic_id][()]
        im = Image.open(io.BytesIO(raw_bytes))
        if im.size != self.target_size:
            im = im.resize(self.target_size, Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.float32)
        return arr

    def close(self):
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None


class ISIC2024Sequence(tf.keras.utils.Sequence):
    """Keras Sequence yielding multi-modal mini-batches (image + tabular metadata)."""

    def __init__(
        self,
        df: pd.DataFrame,
        h5_path: str,
        meta_features: np.ndarray,
        batch_size: int = 32,
        target_size: Tuple[int, int] = (128, 128),
        mode: str = "fusion",  # "fusion", "image_only", "meta_only"
        shuffle: bool = True,
        resample_negatives: bool = False,
        neg_ratio: int = 15,
        seed: Optional[int] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.isic_ids = self.df["isic_id"].values
        self.targets = self.df["target"].values.astype(np.float32) if "target" in self.df.columns else None
        self.meta_features = meta_features.astype(np.float32)
        self.h5_path = str(h5_path)
        self.batch_size = batch_size
        self.target_size = target_size
        self.mode = mode
        self.shuffle = shuffle
        self.resample_negatives = resample_negatives
        self.neg_ratio = neg_ratio
        self.rng = np.random.RandomState(seed)
        self.decoder = HDF5ImageDecoder(h5_path, target_size=target_size)

        if self.resample_negatives and self.targets is not None:
            self.pos_indices = np.where(self.targets == 1.0)[0]
            self.all_neg_indices = np.where(self.targets == 0.0)[0]
            self._resample()
        else:
            self.active_indices = np.arange(len(self.df))
            if self.shuffle:
                self.rng.shuffle(self.active_indices)

    def _resample(self):
        """Draw a fresh negative sample preserving 1:neg_ratio balance and all positives."""
        n_pos = len(self.pos_indices)
        n_neg = n_pos * self.neg_ratio
        sampled_neg = self.rng.choice(
            self.all_neg_indices,
            size=min(n_neg, len(self.all_neg_indices)),
            replace=False,
        )
        self.active_indices = np.concatenate([self.pos_indices, sampled_neg])
        if self.shuffle:
            self.rng.shuffle(self.active_indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.active_indices) / self.batch_size))

    def on_epoch_end(self):
        if self.resample_negatives and self.targets is not None:
            self._resample()
        elif self.shuffle:
            self.rng.shuffle(self.active_indices)

    def __getitem__(self, idx: int):
        batch_idx = self.active_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        b_size = len(batch_idx)

        # 1. Labels
        y_batch = self.targets[batch_idx] if self.targets is not None else None

        # 2. Metadata branch
        meta_batch = self.meta_features[batch_idx]

        if self.mode == "meta_only":
            return meta_batch, y_batch

        # 3. Image branch
        img_batch = np.zeros((b_size, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        for i, b_i in enumerate(batch_idx):
            isic_id = self.isic_ids[b_i]
            img_batch[i] = self.decoder.get_image(isic_id)

        if self.mode == "image_only":
            return img_batch, y_batch

        # Multimodal fusion: inputs as [images, metadata]
        return (img_batch, meta_batch), y_batch
