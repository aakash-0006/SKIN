"""Anomaly detection algorithms for normal hand appearance modeling.

Implements:
1. PCA-Regularized Mahalanobis Distance (Gaussian density baseline)
2. k-Nearest-Neighbor Cosine Distance (Non-parametric baseline)
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class MahalanobisAnomalyDetector:
    """Models normal hand embedding density using PCA and regularized Mahalanobis distance."""

    def __init__(
        self,
        n_components: int = 48,
        regularization: float = 0.05,
        threshold: Optional[float] = None,
    ):
        self.n_components = n_components
        self.regularization = regularization
        self.threshold = threshold
        self.pca: Optional[PCA] = None
        self.mean_vector: Optional[np.ndarray] = None
        self.inv_covariance: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "MahalanobisAnomalyDetector":
        """Fit distribution parameters using training features only.

        Args:
            X_train: Array of shape (N, feature_dim).
        """
        n_samples, n_feats = X_train.shape
        actual_k = min(self.n_components, n_samples - 1, n_feats)
        self.pca = PCA(n_components=actual_k, random_state=42)
        X_pca = self.pca.fit_transform(X_train)

        self.mean_vector = np.mean(X_pca, axis=0)
        cov = np.cov(X_pca, rowvar=False)

        # Ledoit-Wolf-style shrinkage regularization
        trace_cov = np.trace(cov)
        var_scale = trace_cov / actual_k
        cov_reg = (1.0 - self.regularization) * cov + self.regularization * var_scale * np.eye(actual_k)

        self.inv_covariance = np.linalg.pinv(cov_reg)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis anomaly distance for input features."""
        if not self.fitted:
            raise RuntimeError("MahalanobisAnomalyDetector must be fitted before scoring.")
        if X.ndim == 1:
            X = X[None, :]

        X_pca = self.pca.transform(X)
        delta = X_pca - self.mean_vector
        # Vectorized Mahalanobis distance
        left = np.dot(delta, self.inv_covariance)
        dists = np.sqrt(np.maximum(0.0, np.sum(left * delta, axis=1)))
        return dists

    def calibrate_threshold(self, X_val: np.ndarray, alpha: float = 0.05) -> float:
        """Calibrate decision threshold using held-out validation data.

        Args:
            X_val: Held-out normal validation features.
            alpha: Nominal false-rejection rate on normal validation data (default: 5%).

        Returns:
            Calibrated distance threshold.
        """
        val_dists = self.score(X_val)
        self.threshold = float(np.percentile(val_dists, 100.0 * (1.0 - alpha)))
        return self.threshold

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Flag whether samples exceed calibrated anomaly threshold."""
        if self.threshold is None:
            raise RuntimeError("Threshold must be calibrated before calling is_anomaly.")
        scores = self.score(X)
        return scores > self.threshold

    def save(self, filepath: str) -> None:
        """Serialize fitted detector to a compressed .npz archive."""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted detector.")
        np.savez_compressed(
            filepath,
            mean_vector=self.mean_vector,
            inv_covariance=self.inv_covariance,
            pca_components=self.pca.components_,
            pca_mean=self.pca.mean_,
            pca_explained_variance=self.pca.explained_variance_,
            threshold=np.array(self.threshold if self.threshold is not None else -1.0),
            n_components=np.array(self.n_components),
            regularization=np.array(self.regularization),
        )

    @classmethod
    def load(cls, filepath: str) -> "MahalanobisAnomalyDetector":
        """Load fitted detector from .npz archive."""
        data = np.load(filepath)
        detector = cls(
            n_components=int(data["n_components"]),
            regularization=float(data["regularization"]),
            threshold=float(data["threshold"]) if float(data["threshold"]) > 0 else None,
        )
        detector.mean_vector = data["mean_vector"]
        detector.inv_covariance = data["inv_covariance"]

        detector.pca = PCA(n_components=int(data["n_components"]))
        detector.pca.components_ = data["pca_components"]
        detector.pca.mean_ = data["pca_mean"]
        detector.pca.explained_variance_ = data["pca_explained_variance"]

        detector.fitted = True
        return detector


class KNNAnomalyDetector:
    """Models normal hand appearance using k-Nearest-Neighbor distance in feature space."""

    def __init__(self, n_neighbors: int = 5, metric: str = "cosine", threshold: Optional[float] = None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.threshold = threshold
        self.nn: Optional[NearestNeighbors] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "KNNAnomalyDetector":
        """Fit nearest neighbor bank on training data."""
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1)
        self.nn.fit(X_train)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute average distance to k-nearest training neighbors."""
        if not self.fitted:
            raise RuntimeError("KNNAnomalyDetector must be fitted before scoring.")
        if X.ndim == 1:
            X = X[None, :]

        distances, _ = self.nn.kneighbors(X)
        return np.mean(distances, axis=1)

    def calibrate_threshold(self, X_val: np.ndarray, alpha: float = 0.05) -> float:
        """Calibrate decision threshold using held-out validation data."""
        val_dists = self.score(X_val)
        self.threshold = float(np.percentile(val_dists, 100.0 * (1.0 - alpha)))
        return self.threshold

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Threshold must be calibrated before calling is_anomaly.")
        scores = self.score(X)
        return scores > self.threshold
