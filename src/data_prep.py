"""
data_prep.py
------------
Resizes all images to 224x224 and splits them into train / val / test sets
(80% / 10% / 10%).

Expects raw images already downloaded locally into:
    data/raw/Cat/   ← cat images
    data/raw/Dog/   ← dog images

Usage:
    python src/data_prep.py --raw_dir data/raw --output_dir data/processed
"""

import os
import argparse
import random
from PIL import Image

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

TARGET_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def is_valid_image(filepath: str) -> bool:
    """Return True if the file is a readable image."""
    if not filepath.lower().endswith(IMAGE_EXTENSIONS):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def resize_and_save(src_path: str, dst_path: str, size: tuple = TARGET_SIZE):
    """Resize an image to `size` (RGB) and save to `dst_path`."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        img.save(dst_path)


def collect_images(raw_dir: str, label: str) -> list:
    """Return list of image paths for a given label folder inside raw_dir."""
    label_dir = os.path.join(raw_dir, label)
    if not os.path.isdir(label_dir):
        return []
    paths = [
        os.path.join(label_dir, f)
        for f in os.listdir(label_dir)
        if is_valid_image(os.path.join(label_dir, f))
    ]
    return paths


def split_list(items: list, ratios: dict, seed: int = 42) -> dict:
    """Shuffle and split a list into named subsets according to ratios."""
    random.seed(seed)
    shuffled = items[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_dataset(raw_dir: str, output_dir: str):
    """
    Main function: resize all images and write them into
    output_dir/{split}/{label}/ folders.
    """
    labels = ["Cat", "Dog"]

    for label in labels:
        images = collect_images(raw_dir, label)
        if not images:
            print(f"  [WARN] No images found for label '{label}' in {raw_dir}")
            continue

        splits = split_list(images, SPLIT_RATIOS)

        for split, paths in splits.items():
            for src_path in paths:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(output_dir, split, label, filename)
                resize_and_save(src_path, dst_path)

        print(f"  {label}: {len(images)} images split into "
              f"train={len(splits['train'])} / "
              f"val={len(splits['val'])} / "
              f"test={len(splits['test'])}")

    print(f"\nDataset prepared at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Cats vs Dogs dataset")
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw images")
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Path to save processed images"
    )
    args = parser.parse_args()

    prepare_dataset(args.raw_dir, args.output_dir)
