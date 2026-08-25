"""
test_predict.py
---------------
Unit tests for src/predict.py inference utilities.
Run with: pytest tests/test_predict.py
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
from src.predict import preprocess_image, run_inference, LABELS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------- fixtures ----------


@pytest.fixture
def sample_pil_image():
    """Return a plain 100x100 RGB PIL image."""
    return Image.new("RGB", (100, 100), color=(180, 90, 45))


@pytest.fixture
def mock_model():
    """Return a mock Keras model that always predicts dog (0.9)."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.9]])
    return model


@pytest.fixture
def mock_model_cat():
    """Return a mock Keras model that always predicts cat (0.1)."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.1]])
    return model


# ---------- preprocess_image ----------

def test_preprocess_output_shape(sample_pil_image):
    result = preprocess_image(sample_pil_image)
    assert result.shape == (1, 224, 224, 3), f"Expected (1,224,224,3) got {result.shape}"


def test_preprocess_values_in_range(sample_pil_image):
    result = preprocess_image(sample_pil_image)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_handles_rgba_image():
    """RGBA images should be converted to RGB without error."""
    rgba_image = Image.new("RGBA", (50, 50), color=(100, 150, 200, 128))
    result = preprocess_image(rgba_image)
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_dtype_is_float32(sample_pil_image):
    result = preprocess_image(sample_pil_image)
    assert result.dtype == np.float32


# ---------- run_inference ----------

def test_run_inference_returns_label(sample_pil_image, mock_model):
    with patch("src.predict.load_model", return_value=mock_model):
        result = run_inference(sample_pil_image)
    assert "label" in result
    assert result["label"] in LABELS


def test_run_inference_returns_confidence(sample_pil_image, mock_model):
    with patch("src.predict.load_model", return_value=mock_model):
        result = run_inference(sample_pil_image)
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_run_inference_returns_probabilities(sample_pil_image, mock_model):
    with patch("src.predict.load_model", return_value=mock_model):
        result = run_inference(sample_pil_image)
    assert "probabilities" in result
    probs = result["probabilities"]
    assert "cat" in probs and "dog" in probs
    assert abs(probs["cat"] + probs["dog"] - 1.0) < 0.01


def test_run_inference_dog_prediction(sample_pil_image, mock_model):
    """Model output 0.9 should predict dog with high confidence."""
    with patch("src.predict.load_model", return_value=mock_model):
        result = run_inference(sample_pil_image)
    assert result["label"] == "dog"
    assert result["confidence"] > 0.5


def test_run_inference_cat_prediction(sample_pil_image, mock_model_cat):
    """Model output 0.1 should predict cat with high confidence."""
    with patch("src.predict.load_model", return_value=mock_model_cat):
        result = run_inference(sample_pil_image)
    assert result["label"] == "cat"
    assert result["confidence"] > 0.5


def test_load_model_raises_if_missing():
    """Loading a non-existent model path should raise FileNotFoundError."""
    import src.predict as predict_module
    predict_module._model = None  # reset cache
    with pytest.raises(FileNotFoundError):
        predict_module.load_model("nonexistent_model.h5")
