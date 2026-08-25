$ErrorActionPreference = "Stop"

$SERVICE_URL = "http://127.0.0.1:18000"

Write-Host "Service URL: $SERVICE_URL"

Write-Host ""
Write-Host "=== Smoke Test 1: Health Check ==="

$response = Invoke-WebRequest -Uri "$SERVICE_URL/health" -UseBasicParsing

if ($response.StatusCode -ne 200) {
    Write-Host "FAIL: /health returned HTTP $($response.StatusCode)"
    exit 1
}

Write-Host "PASS: /health -> HTTP 200"
Write-Host "Response: $($response.Content)"

Write-Host ""
Write-Host "=== Smoke Test 2: Prediction ==="

$sampleImage = Join-Path $env:TEMP "smoke_test_image.jpg"

python -c @"
from PIL import Image
Image.new('RGB', (224, 224), color=(100, 150, 200)).save(r'$sampleImage')
"@

try {
    $response = curl.exe -s -X POST `
        "$SERVICE_URL/predict" `
        -H "accept: application/json" `
        -F "file=@$sampleImage;type=image/jpeg"

    Write-Host "Response: $response"

    $data = $response | ConvertFrom-Json

    if (-not $data.label) {
        throw "Missing label field"
    }

    if ($data.label -notin @("cat", "dog")) {
        throw "Unexpected label: $($data.label)"
    }

    Write-Host "Predicted label: $($data.label) (confidence: $($data.confidence))"
    Write-Host "PASS: /predict returned a valid prediction"
}
catch {
    Write-Host "FAIL: /predict response is invalid"
    Write-Host $_
    exit 1
}

Write-Host ""
Write-Host "All smoke tests passed. Deployment is healthy."