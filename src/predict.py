"""
predict.py
----------
Inference utility: loads the trained model and runs prediction on a single image.
Used by the FastAPI app and unit tests.
"""

import os
import numpy as np
from PIL import Image
from tensorflow import keras

IMAGE_SIZE = (224, 224)
LABELS = ["cat", "dog"]
_model = None  # module-level cache so the model loads only once


def load_model(model_path: str = "model.h5") -> keras.Model:
    """Load (or return cached) Keras model from disk."""
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _model = keras.models.load_model(model_path)
    return _model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a (1, 224, 224, 3) float32 numpy array
    normalised to [0, 1] — same pipeline used during training.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # add batch dimension


def run_inference(image: Image.Image, model_path: str = "model.h5") -> dict:
    """
    Run prediction on a PIL image.

    Returns:
        {
            "label": "cat" | "dog",
            "confidence": float (0.0 – 1.0),
            "probabilities": {"cat": float, "dog": float}
        }
    """
    model = load_model(model_path)
    input_array = preprocess_image(image)

    raw_prob = float(model.predict(input_array, verbose=0)[0][0])
    dog_prob = raw_prob
    cat_prob = 1.0 - raw_prob

    label = LABELS[int(raw_prob > 0.5)]
    confidence = dog_prob if label == "dog" else cat_prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(cat_prob, 4),
            "dog": round(dog_prob, 4),
        },
    }
