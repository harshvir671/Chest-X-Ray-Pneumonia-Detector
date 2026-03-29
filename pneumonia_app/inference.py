from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "pneumonia_model (2).keras"
MODEL_ZIP = PROJECT_ROOT / "pneumonia_model (2).keras.zip"
CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHTS_PATH = MODEL_DIR / "model.weights.h5"
METADATA_PATH = MODEL_DIR / "metadata.json"
INPUT_SIZE = (224, 224)
THRESHOLD = 0.5

NOTEBOOK_DATASET_STATS = {
    "total_images": 5856,
    "pneumonia_images": 4273,
    "normal_images": 1583,
    "train_images": 4642,
    "test_images": 919,
    "validation_images": 295,
}

PROJECT_REPORTED_METRICS = {
    "validation_accuracy": "92.01%",
    "validation_loss": "0.2316",
    "roc_auc": "~0.90",
    "hybrid_log_loss": "0.106",
}


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    pneumonia_probability: float
    normal_probability: float
    risk_band: str
    threshold: float
    last_conv_layer: str | None
    heatmap: np.ndarray | None


def _import_keras() -> Any:
    try:
        from tensorflow import keras  # type: ignore

        return keras
    except ImportError:
        import keras  # type: ignore

        return keras


def _import_tf() -> Any:
    import tensorflow as tf  # type: ignore

    return tf


@lru_cache(maxsize=1)
def load_model():
    keras = _import_keras()
    errors: list[str] = []

    for candidate in (MODEL_DIR, MODEL_ZIP):
        if not candidate.exists():
            continue
        try:
            return keras.models.load_model(candidate, compile=False)
        except Exception as exc:
            errors.append(f"load_model({candidate.name}) failed: {exc}")

    if not CONFIG_PATH.exists() or not WEIGHTS_PATH.exists():
        message = "Missing Keras bundle files. Expected config.json and model.weights.h5."
        if errors:
            message = f"{message} Previous errors: {' | '.join(errors)}"
        raise FileNotFoundError(message)

    serialized = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    constructors = []

    if hasattr(keras.models, "model_from_json"):
        constructors.append(lambda: keras.models.model_from_json(json.dumps(serialized)))

    saving_api = getattr(keras, "saving", None)
    if saving_api is not None and hasattr(saving_api, "deserialize_keras_object"):
        constructors.append(lambda: saving_api.deserialize_keras_object(serialized))

    for constructor in constructors:
        try:
            model = constructor()
            model.load_weights(WEIGHTS_PATH)
            return model
        except Exception as exc:
            errors.append(f"config.json reconstruction failed: {exc}")

    try:
        model = build_mobilenet_binary_model()
        model.load_weights(WEIGHTS_PATH)
        return model
    except Exception as exc:
        errors.append(f"manual MobileNet reconstruction failed: {exc}")

    raise RuntimeError("Unable to load the Keras model bundle. " + " | ".join(errors))


def build_mobilenet_binary_model():
    keras = _import_keras()
    base_model = keras.applications.MobileNet(
        weights=None,
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3),
    )
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=base_model.input, outputs=outputs)


def preprocess_for_model(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    grayscale = image.convert("L")
    resized = grayscale.resize(INPUT_SIZE, Image.Resampling.BICUBIC)
    rgb = Image.merge("RGB", (resized, resized, resized))
    batch = np.expand_dims(np.asarray(rgb, dtype=np.float32) / 255.0, axis=0)
    return rgb, batch


def prepare_display_image(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    enhanced = ImageOps.autocontrast(grayscale, cutoff=1)
    rgb = enhanced.convert("RGB")
    return ImageOps.contain(rgb, (900, 1100), method=Image.Resampling.LANCZOS)


def predict_image(image: Image.Image, threshold: float = THRESHOLD) -> PredictionResult:
    model = load_model()
    _, batch = preprocess_for_model(image)
    score = float(model.predict(batch, verbose=0).squeeze())
    label = "Pneumonia" if score >= threshold else "Normal"
    confidence = score if label == "Pneumonia" else 1.0 - score
    heatmap, last_conv_layer = make_gradcam_heatmap(batch, model)

    return PredictionResult(
        label=label,
        confidence=float(confidence),
        pneumonia_probability=float(score),
        normal_probability=float(1.0 - score),
        risk_band=_risk_band(score),
        threshold=threshold,
        last_conv_layer=last_conv_layer,
        heatmap=heatmap,
    )


def _risk_band(score: float) -> str:
    if score < 0.2:
        return "Very low pneumonia suspicion"
    if score < 0.5:
        return "Low pneumonia suspicion"
    if score < 0.75:
        return "Moderate pneumonia suspicion"
    return "High pneumonia suspicion"


def make_gradcam_heatmap(batch: np.ndarray, model) -> tuple[np.ndarray | None, str | None]:
    last_conv_layer = find_last_conv_layer_name(model)
    if not last_conv_layer:
        return None, None

    keras = _import_keras()
    tf = _import_tf()
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch, training=False)
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        predictions = tf.convert_to_tensor(predictions)
        target = predictions[:, 0]

    gradients = tape.gradient(target, conv_outputs)
    if gradients is None:
        return None, last_conv_layer

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) <= 0:
        return None, last_conv_layer

    heatmap = heatmap / max_value
    return heatmap.numpy(), last_conv_layer


def find_last_conv_layer_name(model) -> str | None:
    for layer in reversed(model.layers):
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    return None


@lru_cache(maxsize=1)
def get_model_stats() -> dict[str, Any]:
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    layers = config["config"]["layers"]
    dense_layers = [layer for layer in layers if layer["class_name"] == "Dense"]
    output_layer = dense_layers[-1] if dense_layers else {}
    input_shape = layers[0]["config"].get("batch_shape", [None, *INPUT_SIZE, 3])

    stats: dict[str, Any] = {
        "architecture": "MobileNet backbone + GlobalAveragePooling2D + Dense(1, sigmoid)",
        "input_shape": tuple(input_shape[1:]),
        "output_units": output_layer.get("config", {}).get("units", 1),
        "output_activation": output_layer.get("config", {}).get("activation", "sigmoid"),
        "layer_count": len(layers),
        "optimizer": config.get("compile_config", {}).get("optimizer", {}).get("class_name", "Adam"),
        "loss": config.get("compile_config", {}).get("loss", "binary_crossentropy"),
        "metrics": config.get("compile_config", {}).get("metrics", []),
        "saved_keras_version": metadata.get("keras_version", "Unknown"),
        "saved_date": metadata.get("date_saved", "Unknown"),
        "weights_size_mb": round(WEIGHTS_PATH.stat().st_size / (1024 * 1024), 2) if WEIGHTS_PATH.exists() else None,
        "bundle_path": str(MODEL_DIR),
        "dataset": NOTEBOOK_DATASET_STATS,
        "reported_metrics": PROJECT_REPORTED_METRICS,
        "class_map": {"0": "Normal", "1": "Pneumonia"},
    }

    try:
        model = load_model()
        stats["parameter_count"] = int(model.count_params())
        stats["last_conv_layer"] = find_last_conv_layer_name(model)
    except Exception as exc:
        stats["parameter_count"] = None
        stats["last_conv_layer"] = None
        stats["load_warning"] = str(exc)

    return stats
