#!/usr/bin/env bash
# local_test.sh
# -------------
# Builds the Docker image locally, starts the container, and verifies
# both endpoints with curl. Satisfies M2 Task 3b.
#
# Prerequisites: Docker running, model.h5 present in project root.
# Usage: bash scripts/local_test.sh

set -e

IMAGE_NAME="cats-dogs-api:local"
CONTAINER_NAME="cats-dogs-local-test"
PORT=8000
SAMPLE_IMAGE="scripts/sample_cat.jpg"

echo "=== [1/5] Building Docker image ==="
docker build -t "$IMAGE_NAME" .

echo ""
echo "=== [2/5] Starting container ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --name "$CONTAINER_NAME" -p "$PORT:8000" "$IMAGE_NAME"

echo "Waiting for service to start..."
sleep 5

echo ""
echo "=== [3/5] Health check ==="
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health)
if [ "$HEALTH" != "200" ]; then
  echo "FAIL: /health returned HTTP $HEALTH"
  docker logs "$CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
  exit 1
fi
echo "PASS: /health → HTTP 200"
curl -s http://localhost:$PORT/health | python3 -m json.tool

echo ""
echo "=== [4/5] Prediction check ==="
# Create a tiny sample image if no sample exists
if [ ! -f "$SAMPLE_IMAGE" ]; then
  echo "No sample image found at $SAMPLE_IMAGE — generating a blank one for testing."
  python3 -c "
from PIL import Image
import os
os.makedirs('scripts', exist_ok=True)
Image.new('RGB', (224, 224), color=(120, 80, 60)).save('scripts/sample_cat.jpg')
"
fi

PREDICT_RESPONSE=$(curl -s -X POST \
  http://localhost:$PORT/predict \
  -H "accept: application/json" \
  -F "file=@$SAMPLE_IMAGE;type=image/jpeg")

echo "Response: $PREDICT_RESPONSE"

# Check label field exists
if echo "$PREDICT_RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); assert 'label' in d"; then
  echo "PASS: /predict returned a valid response with 'label' field"
else
  echo "FAIL: /predict response missing 'label' field"
  docker rm -f "$CONTAINER_NAME"
  exit 1
fi

echo ""
echo "=== [5/5] Cleanup ==="
docker rm -f "$CONTAINER_NAME"

echo ""
echo "All local tests passed."
