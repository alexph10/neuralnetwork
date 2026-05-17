"""
MNIST data loading utilities.

This module loads the raw MNIST IDX gzip files from:

    data/mnist/raw/

Expected files:
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz

Images are flattened from 28x28 into 784-dimensional vectors and normalized
to the range [0, 1].
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


def _load_idx_images(path: str | Path) -> np.ndarray:
    """
    Load MNIST image data from an IDX gzip file.

    Returns:
        A NumPy array of shape (num_images, 784), dtype float32,
        with pixel values normalized to [0, 1].
    """
    path = Path(path)

    with gzip.open(path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid image file magic number: {magic}")

        buffer = file.read()
        images = np.frombuffer(buffer, dtype=np.uint8)

    images = images.reshape(num_images, rows * cols)
    images = images.astype(np.float32) / 255.0

    return images


def _load_idx_labels(path: str | Path) -> np.ndarray:
    """
    Load MNIST label data from an IDX gzip file.

    Returns:
        A NumPy array of shape (num_labels,), dtype int64.
    """
    path = Path(path)

    with gzip.open(path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid label file magic number: {magic}")

        buffer = file.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    if labels.shape[0] != num_labels:
        raise ValueError(f"Expected {num_labels} labels, but found {labels.shape[0]}")

    return labels.astype(np.int64)


def load_mnist(
    data_dir: str | Path = "data/mnist/raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset.

    Args:
        data_dir:
            Directory containing the raw MNIST IDX gzip files.

    Returns:
        A tuple containing:

        - x_train: shape (60000, 784), dtype float32, values in [0, 1]
        - y_train: shape (60000,), dtype int64
        - x_test: shape (10000, 784), dtype float32, values in [0, 1]
        - y_test: shape (10000,), dtype int64
    """
    data_dir = Path(data_dir)

    train_images_path = data_dir / "train-images-idx3-ubyte.gz"
    train_labels_path = data_dir / "train-labels-idx1-ubyte.gz"
    test_images_path = data_dir / "t10k-images-idx3-ubyte.gz"
    test_labels_path = data_dir / "t10k-labels-idx1-ubyte.gz"

    required_files = [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing MNIST file: {path}")

    x_train = _load_idx_images(train_images_path)
    y_train = _load_idx_labels(train_labels_path)
    x_test = _load_idx_images(test_images_path)
    y_test = _load_idx_labels(test_labels_path)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    print("MNIST loaded successfully.")
    print(
        f"x_train: {x_train.shape}, {x_train.dtype}, min={x_train.min()}, max={x_train.max()}"
    )
    print(f"y_train: {y_train.shape}, {y_train.dtype}")
    print(
        f"x_test:  {x_test.shape}, {x_test.dtype}, min={x_test.min()}, max={x_test.max()}"
    )
    print(f"y_test:  {y_test.shape}, {y_test.dtype}")
