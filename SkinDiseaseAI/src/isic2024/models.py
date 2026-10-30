"""Multimodal, image-only, and metadata-only binary classification models for ISIC 2024."""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@tf.keras.utils.register_keras_serializable(package="src.isic2024")
class BinaryFocalLoss(tf.keras.losses.Loss):
    """Focal Loss for addressing class imbalance with robust Keras serialization."""

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "sum_over_batch_size",
        name: str = "binary_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Compute cross entropy terms
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        loss = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_image_backbone(input_shape=(128, 128, 3), weights="imagenet", trainable: bool = True):
    """Build MobileNetV3Small feature extractor for tile inputs."""
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling="avg",
    )
    base_model.trainable = trainable
    return base_model


def build_multimodal_model(
    image_shape=(128, 128, 3),
    meta_dim: int = 56,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = False,
    backbone_trainable: bool = False,
) -> tf.keras.Model:
    """Construct dual-branch multimodal neural network combining tile images with TBP metadata."""
    # 1. Image branch
    img_input = layers.Input(shape=image_shape, name="image_input")
    backbone = build_image_backbone(input_shape=image_shape, weights="imagenet", trainable=backbone_trainable)
    img_features = backbone(img_input)
    img_dense = layers.Dense(128, activation="relu", name="img_dense")(img_features)
    img_dense = layers.Dropout(0.3, name="img_dropout")(img_dense)

    # 2. Metadata branch
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")
    meta_dense = layers.Dense(64, activation="relu", name="meta_dense_1")(meta_input)
    meta_dense = layers.BatchNormalization(name="meta_bn_1")(meta_dense)
    meta_dense = layers.Dropout(0.2, name="meta_dropout_1")(meta_dense)
    meta_dense = layers.Dense(32, activation="relu", name="meta_dense_2")(meta_dense)

    # 3. Fusion layer
    fused = layers.Concatenate(name="fusion_concat")([img_dense, meta_dense])
    fused = layers.Dense(64, activation="relu", name="fusion_dense")(fused)
    fused = layers.Dropout(0.3, name="fusion_dropout")(fused)
    output = layers.Dense(1, activation="sigmoid", name="malignancy_prob")(fused)

    model = models.Model(inputs=[img_input, meta_input], outputs=output, name="isic2024_multimodal_fusion")

    loss = BinaryFocalLoss(alpha=0.75, gamma=2.0) if use_focal_loss else tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ],
    )
    return model


def unfreeze_upper_backbone(model: tf.keras.Model, unfreeze_layers: int = 30, learning_rate: float = 1e-5):
    """Unfreeze upper layers of the image backbone for Phase 2 fine-tuning."""
    # Find the backbone model inside multimodal or image-only model
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv3" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not locate MobileNetV3 backbone layer in model.")

    backbone.trainable = True
    # Freeze bottom layers and unfreeze only top layers
    if unfreeze_layers > 0:
        freeze_until = max(0, len(backbone.layers) - unfreeze_layers)
        for i, l in enumerate(backbone.layers):
            l.trainable = (i >= freeze_until)

    # Re-compile model with fine-tuning learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=model.loss,
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ],
    )
    return model


def build_image_only_model(
    image_shape=(128, 128, 3),
    learning_rate: float = 1e-4,
    use_focal_loss: bool = False,
    backbone_trainable: bool = True,
) -> tf.keras.Model:
    """Construct baseline image-only classification model."""
    img_input = layers.Input(shape=image_shape, name="image_input")
    backbone = build_image_backbone(input_shape=image_shape, weights="imagenet", trainable=backbone_trainable)
    img_features = backbone(img_input)
    dense = layers.Dense(64, activation="relu", name="img_dense")(img_features)
    dense = layers.Dropout(0.3, name="img_dropout")(dense)
    output = layers.Dense(1, activation="sigmoid", name="malignancy_prob")(dense)

    model = models.Model(inputs=img_input, outputs=output, name="isic2024_image_only_baseline")
    loss = BinaryFocalLoss(alpha=0.75, gamma=2.0) if use_focal_loss else tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def build_meta_only_model(
    meta_dim: int = 56,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = False,
) -> tf.keras.Model:
    """Construct baseline metadata-only tabular MLP classifier."""
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")
    x = layers.Dense(128, activation="relu", name="meta_dense_1")(meta_input)
    x = layers.BatchNormalization(name="meta_bn_1")(x)
    x = layers.Dropout(0.2, name="meta_dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="meta_dense_2")(x)
    x = layers.Dropout(0.2, name="meta_dropout_2")(x)
    x = layers.Dense(32, activation="relu", name="meta_dense_3")(x)
    output = layers.Dense(1, activation="sigmoid", name="malignancy_prob")(x)

    model = models.Model(inputs=meta_input, outputs=output, name="isic2024_metadata_only_baseline")
    loss = BinaryFocalLoss(alpha=0.75, gamma=2.0) if use_focal_loss else tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model
