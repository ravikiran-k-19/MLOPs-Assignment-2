"""
evaluate.py
-----------
Evaluates the trained model on the test set and writes metrics.json.
Called by the DVC 'evaluate' stage.

Usage:
    python src/evaluate.py --data_dir data/processed --model_path model.h5
"""

import os
import json
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)


def evaluate(data_dir: str, model_path: str, batch_size: int = 32):
    model = keras.models.load_model(model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    loss, accuracy = model.evaluate(test_gen, verbose=0)

    metrics = {
        "test_accuracy": round(float(accuracy), 4),
        "test_loss": round(float(loss), 4),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test Accuracy : {metrics['test_accuracy']}")
    print(f"Test Loss     : {metrics['test_loss']}")
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_path", default="model.h5")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_path, args.batch_size)
