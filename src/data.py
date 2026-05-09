from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

def _load_idx_images(path: Path) -> np.ndarray:
    """
    Load and IDX3 image file into an (N, 28, 28) uint8 array.

    Parameters
    ----------
    path : Path
        Path to a gzip-compressed MNIST image file
        (e.g. 'train-images-idx3-ubyte.gz').

    Returns
    -------
    image : np.ndarray
        Array of shape (num_image, 28, 28) with dtype uint8
    """
    with gzip.open(path, "rb") as f:
        # >IIII means: big-endian, 4 unsigned 32-bit integers
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # MNIST magic number for images should be 2051
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file {path}")

        buffer = f.read()
        images = np.frombuffer(buffer, dtype= np.uint8)

    if images.size != num_image * rows * cols:
        raise ValueError(
            f"Unexpected size {path}: "
            f"got {images.size}, expected {nums_images * rows * cols}"
        )
    return images.reshape(num_images, rows, cols)

def _loag_idx_labels(path : Path) -> np.ndarray:
    """
    Load an IDX1 label file into an (N,) uint8 array.

    Parameters
    ----------
    path : Path
        Path to a gzip-compressed MNIST label file
        (e.g. 'train-labels-idx1-ubyte.gz')

    Returns
    -------
    labels: np.ndarray
        Array of shape (num_labels,) with dtype uint8
    """
    with gzip.open(path, "rb") as f:
        # >II means: big-endian, 2 unsigned 32-bit integers
        magic, num_labels = struct.unpack(">II", f.read(8))

        # MNIST magic number for labels should be 2049
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file {path}")

        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    if labels.size != num_labels:
        raise ValueError (
            f"Unexpected size in {path}: got {labels.size}, expected {num_labels}"
        )

    return labels


def load_mnist(
root: str | Path = "data/mnist/raw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset from original IDX .gz files and return model-ready NumPy arrays.

    This function assumes the following files exist under 'root':
        - train-images-idx3-ubyte.gz
        - train-labels-idx3-ubyte.gz
        - train-images-idx1-ubyte.gz
        - train-labels-idx1-ubyte.gz

    Returns:
        X_train : np.ndarray
            Training images as float32 array of shape (10000, 784).
            values normalized to [0, 1]
        y_train : np.ndarray
            Training labels as uint8 array of shape (60000,).
        X_test : np.ndarray
            Test images as float32 array of shape (10000, 784)
            values normalized to [0, 1]
        y_test : np.ndarray
            Test labels as uint8 array of shape (10000, )
    """
    root = Path(root)

    train_images_path = root / "train-images-idx3-ubyte.gz"
    train_labels_path = root / "train-labels-idx1-ubyte.gz"
    test_images_path = root / "t10k-images-idx3-ubyte.gz"
    test_labels_path = root / "t10k-labels-idx1-ubyte.gz"

    # Basic existence checks (fail fast with clear error)
    for p in (train_images_path, train_labels_path, test_images_path, test_labels_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Required MNIST file not found: {p}. "
                "Ensure you placed the original .gz files under data/mnist/raw."
            )

    # Load raw uint8 images and labels
    X_train_raw = _load_idx_images(train_images_path)
    y_train = _load_idx_labels(train_labels_path)
    X_test_raw = _load_idx_images(test_images_path)
    y_test = _load_idx_labels(test_labels_path)

    X_train = X_train_raw.astype(np.float32)
    X_test = X_test_raw.astype(np.float32)

    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_test, y_test

if __name__ = "__main__":
    # Quick sanity check when running file
    X_train, y_train, X_test, y_test = load_mnist()

    print("X_train:", X_train.shape, X_train.dtype, X_train.min(), X_train.max())
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype, X_test.min(), X_test.max())
    print("y_test:", y_test.shape, y_test.dtype)
