"""
train.py
--------
Trains the CNN on processed data and tracks the experiment with MLflow.

Logs to MLflow:
    - Parameters : epochs, batch_size, learning_rate, image_size
    - Metrics    : train/val accuracy and loss per epoch
    - Artifacts  : loss curve plot, confusion matrix, saved model (.h5)

Usage:
    python src/train.py --data_dir data/processed --epochs 10 --batch_size 32
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from model import build_model

# Label mapping used consistently across train / predict
LABELS = ["cat", "dog"]  # index 0 = cat, index 1 = dog
IMAGE_SIZE = (224, 224)
MODEL_PATH = "model.h5"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def make_generators(data_dir: str, batch_size: int):
    """Create train, val, and test data generators with augmentation on train."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def plot_loss_curve(history, save_path: str):
    """Save training vs validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(model, test_gen, save_path: str):
    """Generate and save confusion matrix on test set."""
    test_gen.reset()
    preds = (model.predict(test_gen) > 0.5).astype(int).flatten()
    true_labels = test_gen.classes

    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(data_dir: str, epochs: int, batch_size: int, learning_rate: float):
    
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("cats-vs-dogs-v2")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.set_tag("mlflow.user", "Ravikiran")
        mlflow.set_tag("mlflow.runName", "Cats-Dogs-Baseline-v1")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("image_size", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

        train_gen, val_gen, test_gen = make_generators(data_dir, batch_size)

        model = build_model(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            learning_rate=learning_rate,
        )
        model.summary()

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
        )

        # Log per-epoch metrics
        for epoch_idx in range(epochs):
            mlflow.log_metric("train_loss", history.history["loss"][epoch_idx], step=epoch_idx)
            mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch_idx], step=epoch_idx)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch_idx], step=epoch_idx)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch_idx], step=epoch_idx)

        # Evaluate on test set
        test_gen.reset()
        test_loss, test_acc = model.evaluate(test_gen)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

        # Save artifacts
        loss_curve_path = "loss_curve.png"
        confusion_matrix_path = "confusion_matrix.png"

        plot_loss_curve(history, loss_curve_path)
        plot_confusion_matrix(model, test_gen, confusion_matrix_path)

        mlflow.log_artifact(loss_curve_path)
        mlflow.log_artifact(confusion_matrix_path)

        # Save and log the model
        model.save(MODEL_PATH)
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"Model saved to {MODEL_PATH}")

        # Clean up temp plot files
        os.remove(loss_curve_path)
        os.remove(confusion_matrix_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs CNN")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.learning_rate)
