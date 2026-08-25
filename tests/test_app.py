import json
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import api.app as app_module


client = TestClient(app_module.app)


def reset_stats():
    """Reset application statistics between tests."""
    app_module._stats.update(
        {
            "total_requests": 0,
            "cat_count": 0,
            "dog_count": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
        }
    )


def create_test_image():
    """Create a small valid JPEG image for upload tests."""
    image = Image.new("RGB", (224, 224), (100, 150, 200))

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "model_path" in data


def test_metrics_initial_state():
    reset_stats()

    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_requests"] == 0
    assert data["cat_count"] == 0
    assert data["dog_count"] == 0
    assert data["errors"] == 0
    assert data["avg_latency_ms"] == 0.0


def test_predict_cat(monkeypatch):
    reset_stats()

    def mock_inference(image, model_path):
        return {
            "label": "cat",
            "confidence": 0.95,
            "probabilities": {
                "cat": 0.95,
                "dog": 0.05,
            },
        }

    monkeypatch.setattr(app_module, "run_inference", mock_inference)

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "cat"
    assert data["confidence"] == 0.95
    assert data["probabilities"]["cat"] == 0.95

    assert app_module._stats["total_requests"] == 1
    assert app_module._stats["cat_count"] == 1


def test_predict_dog(monkeypatch):
    reset_stats()

    def mock_inference(image, model_path):
        return {
            "label": "dog",
            "confidence": 0.90,
            "probabilities": {
                "cat": 0.10,
                "dog": 0.90,
            },
        }

    monkeypatch.setattr(app_module, "run_inference", mock_inference)

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "dog"
    assert data["confidence"] == 0.90

    assert app_module._stats["total_requests"] == 1
    assert app_module._stats["dog_count"] == 1


def test_predict_rejects_invalid_file_type():
    reset_stats()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only JPEG and PNG images are supported."
    )

    assert app_module._stats["errors"] == 1


def test_predict_rejects_corrupted_image():
    reset_stats()

    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupt.jpg",
                b"this is not a valid image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Could not read the uploaded image."
    )

    assert app_module._stats["errors"] == 1


def test_predict_model_not_found(monkeypatch):
    reset_stats()

    def mock_inference(image, model_path):
        raise FileNotFoundError("Model file not found")

    monkeypatch.setattr(app_module, "run_inference", mock_inference)

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model file not found"
    assert app_module._stats["errors"] == 1


def test_predict_inference_error(monkeypatch):
    reset_stats()

    def mock_inference(image, model_path):
        raise RuntimeError("Inference failed")

    monkeypatch.setattr(app_module, "run_inference", mock_inference)

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Inference failed."
    assert app_module._stats["errors"] == 1


def test_metrics_after_predictions(monkeypatch):
    reset_stats()

    def mock_inference(image, model_path):
        return {
            "label": "cat",
            "confidence": 0.85,
            "probabilities": {
                "cat": 0.85,
                "dog": 0.15,
            },
        }

    monkeypatch.setattr(app_module, "run_inference", mock_inference)

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_requests"] == 1
    assert data["cat_count"] == 1
    assert data["dog_count"] == 0
    assert data["errors"] == 0
    assert data["avg_latency_ms"] >= 0


def test_performance_when_log_does_not_exist(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "PERFORMANCE_LOG",
        "/tmp/nonexistent-performance-log.json",
    )

    response = client.get("/performance")

    assert response.status_code == 404
    assert "No performance log found" in response.json()["detail"]


def test_performance_returns_log(tmp_path, monkeypatch):
    performance_file = tmp_path / "performance_log.json"

    performance_data = {
        "timestamp": "2026-08-25T00:00:00Z",
        "api_url": "http://localhost:8000",
        "total_samples": 2,
        "correct": 2,
        "accuracy": 1.0,
        "results": [],
    }

    performance_file.write_text(
        json.dumps(performance_data),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app_module,
        "PERFORMANCE_LOG",
        str(performance_file),
    )

    response = client.get("/performance")

    assert response.status_code == 200
    assert response.json() == performance_data
