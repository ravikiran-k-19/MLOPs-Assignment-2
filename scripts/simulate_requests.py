"""
simulate_requests.py
--------------------
Post-deployment performance tracker (M5).

Sends a batch of test images (with known true labels) to the deployed API,
computes accuracy, and writes a performance_log.json in the project root.
The /performance endpoint in app.py reads this file.

Usage:
    python scripts/simulate_requests.py \
        --api_url http://localhost:8000 \
        --test_dir data/processed/test \
        --max_samples 20
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime


def send_prediction(api_url: str, image_path: str) -> dict | None:
    """Send a single image to POST /predict and return the JSON response."""
    with open(image_path, "rb") as f:
        try:
            response = requests.post(
                f"{api_url}/predict",
                files={"file": (os.path.basename(image_path), f, "image/jpeg")},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  [ERROR] {image_path}: {e}")
            return None


def collect_test_images(test_dir: str, max_samples: int) -> list:
    """
    Walk test_dir/{cats,dogs}/ and return up to max_samples
    (image_path, true_label) tuples.
    """
    samples = []
    for folder, label in [("Cat", "cat"), ("Dog", "dog")]:
        label_dir = os.path.join(test_dir, folder)
        if not os.path.isdir(label_dir):
            print(f"[WARN] {label_dir} not found — skipping label '{folder}'")
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(label_dir, fname), label))
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break
    return samples[:max_samples]


def run_simulation(api_url: str, test_dir: str, max_samples: int):
    print(f"Target API : {api_url}")
    print(f"Test dir   : {test_dir}")
    print(f"Samples    : {max_samples}")
    print("")

    samples = collect_test_images(test_dir, max_samples)
    if not samples:
        print("No test images found. Exiting.")
        sys.exit(1)

    results = []
    correct = 0

    for idx, (image_path, true_label) in enumerate(samples, 1):
        print(
            f"[{idx}/{len(samples)}] {os.path.basename(image_path)}"
            f"(true: {true_label})", 
            end=" → ",
            )
        response = send_prediction(api_url, image_path)

        if response is None:
            print("ERROR")
            continue

        predicted = response.get("label", "unknown")
        confidence = response.get("confidence", 0.0)
        is_correct = predicted == true_label

        if is_correct:
            correct += 1

        print(f"predicted: {predicted} ({confidence:.2f}) | {'✓' if is_correct else '✗'}")

        results.append({
            "image": os.path.basename(image_path),
            "true_label": true_label,
            "predicted_label": predicted,
            "confidence": confidence,
            "correct": is_correct,
        })

    total = len(results)
    accuracy = round(correct / total, 4) if total > 0 else 0.0

    print(f"\nAccuracy: {correct}/{total} = {accuracy * 100:.1f}%")

    log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "api_url": api_url,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    log_path = os.path.join(os.path.dirname(__file__), "..", "performance_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Performance log saved to: {os.path.abspath(log_path)}")
    print("View via: GET /performance on the running API")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate post-deployment requests and measure accuracy"
    )

    parser.add_argument(
        "--api_url",
        default="http://localhost:8000",
        help="Base URL of the deployed API",
    )
    
    parser.add_argument(
        "--test_dir",
        default="data/processed/test",
        help="Path to test set directory",
    )
    
    parser.add_argument("--max_samples", type=int, default=20, help="Max number of images to send")
    args = parser.parse_args()

    run_simulation(args.api_url, args.test_dir, args.max_samples)
