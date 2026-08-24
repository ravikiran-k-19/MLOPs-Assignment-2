# Cats vs Dogs MLOps Pipeline

Binary image classification (Cats vs Dogs) — end-to-end MLOps pipeline built for BITS Pilani MLOps Assignment 2.

## Project Structure

```
cats-vs-dogs-mlops/
├── data/                        # DVC-tracked dataset
│   ├── raw/                     # Original Kaggle images
│   └── processed/               # Resized 224x224 train/val/test splits
├── src/
│   ├── data_prep.py             # Download, resize, split dataset
│   ├── model.py                 # Keras CNN definition
│   ├── train.py                 # Training + MLflow experiment tracking
│   └── predict.py               # Inference utility used by the API
├── api/
│   └── app.py                   # FastAPI service (health, predict, metrics, performance)
├── tests/
│   ├── test_data_prep.py        # Unit tests for preprocessing
│   └── test_predict.py          # Unit tests for inference
├── deployment/
│   ├── k8s/                     # Raw Kubernetes manifests
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── helm/                    # Helm chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/
│   └── smoke_test.sh            # Post-deploy smoke test
├── scripts/
│   ├── local_test.sh            # Build + run Docker locally and verify
│   └── simulate_requests.py     # Post-deploy performance tracking
├── .github/workflows/
│   ├── ci.yml                   # CI: test → build → push image
│   └── cd.yml                   # CD: deploy via Helm on minikube
├── Dockerfile
├── requirements.txt
└── dvc.yaml                     # DVC pipeline stages
```

## Milestones

| Milestone | Description |
|-----------|-------------|
| M1 | Model development + experiment tracking (DVC + MLflow) |
| M2 | Model packaging + containerization (FastAPI + Docker) |
| M3 | CI pipeline (GitHub Actions + Docker Hub) |
| M4 | CD pipeline + Kubernetes deployment (minikube + Helm) |
| M5 | Monitoring, logging, performance tracking |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
dvc repro prepare
```

### 3. Train model
```bash
dvc repro train
```

### 4. View MLflow experiments
```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Run API locally
```bash
uvicorn api.app:app --reload --port 8000
```

### 6. Run tests
```bash
pytest tests/
```

### 7. Build and test Docker image locally
```bash
bash scripts/local_test.sh
```

### 8. Deploy to minikube
```bash
helm upgrade --install cats-dogs-app deployment/helm
bash deployment/smoke_test.sh
```
