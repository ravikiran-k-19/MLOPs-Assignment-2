#!/usr/bin/env bash
# smoke_test.sh
# -------------
# Post-deployment smoke test for M4.
# Calls /health and /predict on the running service.
# Exits non-zero on any failure — this fails the CD pipeline.
#
# Usage: bash deployment/smoke_test.sh

set -e

# Resolve the service URL from minikube
SERVICE_URL=$(minikube service cats-dogs-app --url 2>/dev/null || echo "http://localhost:30080")
echo "Service URL: $SERVICE_URL"

echo ""
echo "=== Smoke Test 1: Health Check ==="
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health")

if [ "$HTTP_STATUS" != "200" ]; then
  echo "FAIL: /health returned HTTP $HTTP_STATUS (expected 200)"
  exit 1
fi

HEALTH_BODY=$(curl -s "$SERVICE_URL/health")
echo "PASS: /health → HTTP 200"
echo "Response: $HEALTH_BODY"

echo ""
echo "=== Smoke Test 2: Prediction ==="

# Generate a small test image inline if no sample exists
SAMPLE_IMAGE="/tmp/smoke_test_image.jpg"
python3 -c "
from PIL import Image
Image.new('RGB', (224, 224), color=(100, 150, 200)).save('$SAMPLE_IMAGE')
"

PREDICT_RESPONSE=$(curl -s -X POST \
  "$SERVICE_URL/predict" \
  -H "accept: application/json" \
  -F "file=@$SAMPLE_IMAGE;type=image/jpeg")

echo "Response: $PREDICT_RESPONSE"

# Validate that 'label' field is present in the response
python3 -c "
import sys, json
data = json.loads('$PREDICT_RESPONSE')
assert 'label' in data, 'Missing label field'
assert data['label'] in ['cat', 'dog'], f'Unexpected label: {data[\"label\"]}'
print(f'Predicted label: {data[\"label\"]} (confidence: {data[\"confidence\"]})')
" || {
  echo "FAIL: /predict response is invalid"
  exit 1
}

echo "PASS: /predict returned a valid prediction"

echo ""
echo "All smoke tests passed. Deployment is healthy."
