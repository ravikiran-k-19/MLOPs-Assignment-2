"""
model.py
--------
Defines the baseline Keras CNN for binary image classification (Cats vs Dogs).

Architecture:
    3x [Conv2D -> BatchNorm -> MaxPool] blocks
    Flatten -> Dense(128) -> Dropout -> Dense(1, sigmoid)

Input:  (224, 224, 3)
Output: scalar probability (>0.5 = dog, <=0.5 = cat)
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple = (224, 224, 3), learning_rate: float = 0.001) -> keras.Model:
    """Build and compile the CNN model."""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="cats_vs_dogs_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
