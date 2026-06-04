from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from data import load_mnist
from model import NeuralNet


Array = np.ndarray


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-2
    seed: int = 42


@dataclass
class TrainingHistory:
    train_loss: List[float]
    train_accuracy: List[float]
    test_accuracy: List[float]


def softmax_cross_entropy_loss(logits: Array, y_true: Array) -> Tuple[float, Array]:
    """
    Compute the softmax cross-entropy loss and its gradient w.r.t. logits.

    Parameters
    logits : np.ndarray
        Array of shape (batch_size, num_classes) containing raw model outputs.
    y_true : np.ndarray
        Array of shape (batch_size,) containing integer class labels in
        the range [0, num_classes - 1].

    Returns
    loss : float
        Scalar cross-entropy loss averaged over the batch.
    dlogits : np.ndarray
        Array of shape (batch_size, num_classes) with the gradient of the loss
        w.r.t. the logits.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2D (batch_size, num_classes), got {logits.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D (batch_size,), got {y_true.shape}")
    if logits.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Batch size mismatch between logits and y_true: "
            f"{logits.shape[0]} vs {y_true.shape[0]}"
        )

    batch_size, num_classes = logits.shape

    # Numerically stable softmax: subtract max per row before exponentiating.
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Cross-entropy loss: -log p(correct_class), averaged over batch.
    # Clip probabilities to avoid log(0).
    eps = 1e-15
    clipped = np.clip(probs[np.arange(batch_size), y_true], eps, 1.0)
    loss = -np.mean(np.log(clipped))

    # Gradient of loss w.r.t logits for softmax + cross-entropy:
    # dL/dz = p - y_one_hot, averaged over batch.
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), y_true] -= 1.0
    dlogits /= batch_size

    return loss, dlogits


def compute_accuracy(model: NeuralNet, X: Array, y: Array) -> float:
    """
    Compute the classification accuracy of a model on a dataset.

    Parameters
    model : NeuralNet
        Trained or partially trained model.
    X : np.ndarray
        Input data of shape (num_samples, input_dim).
    y : np.ndarray
        Integer labels of shape (num_samples,).

    Returns
    accuracy : float
        Fraction of correctly classified examples in [0.0, 1.0].
    """
    logits = model.forward(X)
    y_pred = np.argmax(logits, axis=1)
    return float(np.mean(y_pred == y))


def iterate_minibatches(
    X: Array,
    y: Array,
    batch_size: int,
    *,
    shuffle: bool = True,
    rng: np.random.Generator,
):
    """
    Generate mini-batches of (X_batch, y_batch) from the full dataset.

    Parameters
    X : np.ndarray
        Input data of shape (num_samples, input_dim).
    y : np.ndarray
        Labels of shape (num_samples,).
    batch_size : int
        Number of samples per mini-batch.
    shuffle : bool, optional
        Whether to shuffle the data before creating batches.
    rng : np.random.Generator
        Random number generator for reproducible shuffling.

    Yields
    X_batch : np.ndarray
    y_batch : np.ndarray
    """
    num_samples = X.shape[0]
    if shuffle:
        indices = rng.permutation(num_samples)
    else:
        indices = np.arange(num_samples)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


def train(config: TrainingConfig) -> Tuple[NeuralNet, TrainingHistory]:
    """
    Train a NeuralNet on the MNIST dataset using mini-batch SGD.

    Parameters
    config : TrainingConfig
        Hyperparameters for the training run.

    Returns
    model : NeuralNet
        Trained model instance.
    history : TrainingHistory
        Object containing per-epoch loss and accuracy metrics.
    """
    rng = np.random.default_rng(config.seed)

    # Load data
    X_train, y_train, X_test, y_test = load_mnist()

    model = NeuralNet()

    history = TrainingHistory(
        train_loss=[],
        train_accuracy=[],
        test_accuracy=[],
    )

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch SGD
        for X_batch, y_batch in iterate_minibatches(
            X_train, y_train, batch_size=config.batch_size, shuffle=True, rng=rng
        ):
            logits = model.forward(X_batch)
            loss, dlogits = softmax_cross_entropy_loss(logits, y_batch)

            grads: Dict[str, List[Array]] = model.backward(dlogits)
            model.update_params(grads, learning_rate=config.learning_rate)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        train_acc = compute_accuracy(model, X_train, y_train)
        test_acc = compute_accuracy(model, X_test, y_test)

        history.train_loss.append(avg_loss)
        history.train_accuracy.append(train_acc)
        history.test_accuracy.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} "
            f"- loss: {avg_loss:.4f} "
            f"- train_acc: {train_acc:.4f} "
            f"- test_acc: {test_acc:.4f}"
        )

    return model, history
