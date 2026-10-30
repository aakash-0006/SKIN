"""Preprocessing shared by training and inference.

EfficientNetB0 in tf.keras.applications already contains its own rescaling/normalization
layers and expects raw pixel values in [0, 255], so we ONLY resize and keep 0-255 floats.
"""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

MIN_SIDE = 64
MAX_BYTES = 15 * 1024 * 1024


def _load_and_resize(path, label, img_size):
    import tensorflow as tf
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img.set_shape((img_size, img_size, 3))
    return tf.cast(img, tf.float32), label


def make_dataset(paths, labels, img_size, batch_size, training=False, seed=42):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, dtype=np.int32)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _load_and_resize(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def pil_to_array(img, img_size):
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]  # 0-255, same as training


def build_augmentation_layer(
    rotation_degrees=20.0,
    zoom_pct=0.08,
    translation_pct=0.05,
    name="augmentation",
):
    """Conservative, medically appropriate online data augmentation.

    Transforms:
    - Horizontal flip (skin lesions have no canonical left/right orientation; vertical flip is avoided).
    - Random rotation up to ±20 degrees (factor = 20 / 360).
    - Random zoom 5-10% (factor = (-0.08, 0.08)).
    - Random translation ~5% (factor = (-0.05, 0.05) horizontally and vertically).
    - Maintains [0, 255] float32 scale and 224x224 dimensions for EfficientNetB0.
    - Active only during training=True; inactive (exact identity) during validation/test.
    """
    import tensorflow as tf
    rot_factor = float(rotation_degrees) / 360.0
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal", name="aug_hflip"),
        tf.keras.layers.RandomRotation(
            factor=(-rot_factor, rot_factor),
            fill_mode="reflect",
            interpolation="bilinear",
            name="aug_rotation",
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-zoom_pct, zoom_pct),
            fill_mode="reflect",
            interpolation="bilinear",
            name="aug_zoom",
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=(-translation_pct, translation_pct),
            width_factor=(-translation_pct, translation_pct),
            fill_mode="reflect",
            interpolation="bilinear",
            name="aug_translation",
        ),
    ], name=name)


def apply_augmentation_to_model(source_model, aug_layer=None):
    """Rebuild model with the conservative augmentation layer, preserving all trained weights.

    Replaces any existing 'augmentation' layer in source_model with aug_layer while
    reusing 'efficientnetb0', 'global_average_pooling2d', 'dropout', and 'dense' layers.
    """
    import tensorflow as tf
    if aug_layer is None:
        aug_layer = build_augmentation_layer()

    inp = tf.keras.layers.Input(shape=source_model.input_shape[1:], name=source_model.layers[0].name)
    x = aug_layer(inp)

    # Chain through the downstream layers (efficientnetb0, global_average_pooling2d, dropout, dense)
    for layer in source_model.layers[1:]:
        if layer.name == "augmentation":
            continue
        x = layer(x)

    new_model = tf.keras.Model(inputs=inp, outputs=x, name=source_model.name)
    return new_model


def read_image_bytes(data):
    """Validate raw bytes. Returns (PIL image or None, blocking errors, warnings)."""
    warnings = []
    if not data:
        return None, ["The file is empty."], warnings
    if len(data) > MAX_BYTES:
        return None, ["The file is larger than 15 MB."], warnings
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None, ["The file is not a readable JPG/PNG image."], warnings
    if img.format not in ("JPEG", "PNG"):
        return None, [f"Unsupported format: {img.format}. Use JPG or PNG."], warnings
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        return None, [f"Image is too small ({w}x{h}px). Minimum side is {MIN_SIDE}px."], warnings
    if min(w, h) < 224:
        warnings.append(f"Low resolution ({w}x{h}px); the image will be upscaled and results may be unreliable.")
    return img, [], warnings


def quality_warnings(img):
    """Cheap heuristics: brightness, contrast, blur (variance of a Laplacian filter)."""
    w = []
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    if g.mean() < 40:
        w.append("The image is very dark.")
    elif g.mean() > 225:
        w.append("The image is overexposed.")
    if g.std() < 12:
        w.append("The image has very low contrast.")
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    if lap.var() < 20:
        w.append("The image appears blurry or out of focus.")
    return w
