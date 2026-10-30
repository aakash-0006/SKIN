"""Pretrained CPU-compatible feature extractor for hand images.

Uses frozen MobileNetV3Small (ImageNet weights, global average pooling)
to extract compact 576-dimensional semantic appearance embeddings.
"""
from typing import List, Union
import numpy as np
import tensorflow as tf


class HandFeatureExtractor:
    """Extracts compact feature embeddings using a frozen MobileNetV3Small backbone."""

    def __init__(self, target_size: tuple = (224, 224)):
        self.target_size = target_size
        self._model = tf.keras.applications.MobileNetV3Small(
            input_shape=(target_size[0], target_size[1], 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        self._model.trainable = False
        self.feature_dim = 576

    def extract_features(
        self,
        images_bgr: Union[np.ndarray, List[np.ndarray]],
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract deep feature representations for a batch of BGR hand crops.

        Args:
            images_bgr: Single BGR array (H, W, 3) or list/batch of arrays (N, H, W, 3).
            normalize: If True, L2-normalize the resulting feature vectors.

        Returns:
            numpy array of shape (N, 576).
        """
        if isinstance(images_bgr, np.ndarray) and images_bgr.ndim == 3:
            images_bgr = [images_bgr]

        # Convert BGR to RGB and convert to float32
        batch_rgb = []
        for img in images_bgr:
            if img.shape[:2] != self.target_size:
                img = tf.image.resize(img, self.target_size).numpy().astype(np.uint8)
            rgb = img[..., ::-1]  # BGR to RGB
            batch_rgb.append(rgb)

        arr = np.asarray(batch_rgb, dtype=np.float32)
        # MobileNetV3 preprocessing: maps [0, 255] to [-1, 1]
        preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(arr)

        features = self._model(preprocessed, training=False).numpy()

        if normalize:
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            features = features / norms

        return features
