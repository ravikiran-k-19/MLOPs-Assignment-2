"""
test_data_prep.py
-----------------
Unit tests for src/data_prep.py preprocessing functions.
Run with: pytest tests/test_data_prep.py
"""

import os
import sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_prep import is_valid_image, resize_and_save, split_list, collect_images


# ---------- fixtures ----------

@pytest.fixture
def sample_image(tmp_path):
    """Create a small valid RGB JPEG for testing."""
    img_path = tmp_path / "cat.jpg"
    img = Image.new("RGB", (100, 100), color=(120, 60, 30))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a small folder with 10 JPEG images."""
    label_dir = tmp_path / "Cat"
    label_dir.mkdir()
    for i in range(10):
        img = Image.new("RGB", (50, 50), color=(i * 20, 0, 0))
        img.save(label_dir / f"cat_{i}.jpg")
    return str(tmp_path), "Cat"


# ---------- is_valid_image ----------

def test_valid_image_returns_true(sample_image):
    assert is_valid_image(sample_image) is True


def test_non_image_extension_returns_false(tmp_path):
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("not an image")
    assert is_valid_image(str(txt_file)) is False


def test_corrupted_image_returns_false(tmp_path):
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"not image data")
    assert is_valid_image(str(bad_file)) is False


# ---------- resize_and_save ----------

def test_resize_output_is_224x224(sample_image, tmp_path):
    dst = str(tmp_path / "resized.jpg")
    resize_and_save(sample_image, dst, size=(224, 224))
    with Image.open(dst) as img:
        assert img.size == (224, 224)


def test_resize_output_is_rgb(sample_image, tmp_path):
    dst = str(tmp_path / "resized.jpg")
    resize_and_save(sample_image, dst)
    with Image.open(dst) as img:
        assert img.mode == "RGB"


def test_resize_creates_parent_dirs(sample_image, tmp_path):
    dst = str(tmp_path / "nested" / "dir" / "output.jpg")
    resize_and_save(sample_image, dst)
    assert os.path.exists(dst)


# ---------- split_list ----------

def test_split_list_lengths_sum_to_total():
    items = list(range(100))
    splits = split_list(items, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 100


def test_split_list_train_is_80_percent():
    items = list(range(100))
    splits = split_list(items, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert len(splits["train"]) == 80


def test_split_list_is_reproducible():
    items = list(range(50))
    s1 = split_list(items, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    s2 = split_list(items, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    assert s1["train"] == s2["train"]


def test_split_list_no_overlap():
    items = list(range(100))
    splits = split_list(items, {"train": 0.8, "val": 0.1, "test": 0.1})
    all_items = splits["train"] + splits["val"] + splits["test"]
    assert len(all_items) == len(set(all_items))


# ---------- collect_images ----------

def test_collect_images_returns_correct_count(sample_image_dir):
    raw_dir, label = sample_image_dir
    paths = collect_images(raw_dir, label)
    assert len(paths) == 10


def test_collect_images_missing_label_returns_empty(tmp_path):
    paths = collect_images(str(tmp_path), "nonexistent")
    assert paths == []
