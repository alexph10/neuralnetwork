from __future__ import annotations

import numpy as np

Array = np.ndarray
Gradients = dict[str, list[Array]]


def relu(x: Array) -> Array:
    """
    Apply the ReLU activation function elementwise.

    Args:
        x:
            Input array.

    Returns:
        An array with negative values replaced by zero.
    """
    return np.maximum(0, x)


def relu_derivative(x: Array) -> Array:
    """
    Compute the derivative of ReLU with respect to its input.

    Args:
        x:
            Pre-activation values.

    Returns:
        An array containing 1 where x > 0 and 0 elsewhere.
    """
    return (x > 0).astype(np.float32)


def softmax(logits: Array) -> Array:
    """
    Convert logits into probabilities using a numerically stable softmax.

    Args:
        logits:
            Raw class scores with shape (batch_size, num_classes).

    Returns:
        Class probabilities with shape (batch_size, num_classes).
    """
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted_logits)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return probabilities


class NeuralNet:
    """
    A fully connected multilayer perceptron for MNIST classification.

    Args:
        input_dim:
            Number of input features.
            For flattened MNIST images, this is 784.

        hidden_dims:
            Hidden-layer widths.

        output_dim:
            Number of output classes.
            For MNIST, this is 10.

        weight_scale:
            Standard deviation multiplier used for random weight initialization.

        rng:
            Optional NumPy random generator for reproducible initialization.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        output_dim: int = 10,
        weight_scale: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.layer_dims = [input_dim, *hidden_dims, output_dim]
        self.num_layers = len(self.layer_dims) - 1

        self.rng = rng if rng is not None else np.random.default_rng()

        self.weights: list[Array] = []
        self.biases: list[Array] = []

        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            weight = weight_scale * self.rng.standard_normal(
                size=(in_dim, out_dim),
                dtype=np.float32,
            )
            bias = np.zeros((1, out_dim), dtype=np.float32)

            self.weights.append(weight)
            self.biases.append(bias)

        self._cache: dict[str, list[Array] | Array] = {}

    def forward(self, X: Array, apply_softmax: bool = False) -> Array:
        """
        Run a forward pass through the network.

        Args:
            X:
                Input batch with shape (batch_size, input_dim).

            apply_softmax:
                If True, return class probabilities instead of raw logits.

        Returns:
            Raw logits if apply_softmax=False.
            Class probabilities if apply_softmax=True.

        Notes:
            This method stores intermediate values in self._cache for backward().
        """
        if X.ndim != 2:
            raise ValueError(
                f"Expected X to have shape (batch_size, input_dim), got {X.shape}"
            )

        if X.shape[1] != self.layer_dims[0]:
            raise ValueError(
                f"Expected input dimension {self.layer_dims[0]}, got {X.shape[1]}"
            )

        activations: list[Array] = [X]
        pre_activations: list[Array] = []

        A = X

        for layer_idx in range(self.num_layers - 1):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]

            Z = A @ W + b
            A = relu(Z)

            pre_activations.append(Z)
            activations.append(A)

        W_out = self.weights[-1]
        b_out = self.biases[-1]
        logits = A @ W_out + b_out

        self._cache = {
            "activations": activations,
            "pre_activations": pre_activations,
            "logits": logits,
        }

        if apply_softmax:
            return softmax(logits)

        return logits

    def backward(self, dlogits: Array) -> Gradients:
        """
        Run backpropagation through the network.

        Args:
            dlogits:
                Gradient of the loss with respect to the output logits.
                Shape: (batch_size, output_dim).

        Returns:
            A dictionary containing:

            - "weights": gradients for each weight matrix
            - "biases": gradients for each bias vector
        """
        if not self._cache:
            raise ValueError("No cache found. Run forward() before backward().")

        activations = self._cache["activations"]
        pre_activations = self._cache["pre_activations"]

        if not isinstance(activations, list):
            raise TypeError("self._cache['activations'] must be a list.")

        if not isinstance(pre_activations, list):
            raise TypeError("self._cache['pre_activations'] must be a list.")

        if dlogits.ndim != 2:
            raise ValueError(
                f"Expected dlogits to have shape (batch_size, output_dim), "
                f"got {dlogits.shape}"
            )

        expected_output_dim = self.layer_dims[-1]
        if dlogits.shape[1] != expected_output_dim:
            raise ValueError(
                f"Expected dlogits second dimension to be {expected_output_dim}, "
                f"got {dlogits.shape[1]}"
            )

        grad_weights: list[Array] = [np.zeros_like(weight) for weight in self.weights]
        grad_biases: list[Array] = [np.zeros_like(bias) for bias in self.biases]

        grad = dlogits

        for layer_idx in reversed(range(self.num_layers)):
            previous_activation = activations[layer_idx]

            grad_weights[layer_idx] = previous_activation.T @ grad
            grad_biases[layer_idx] = np.sum(grad, axis=0, keepdims=True)

            if layer_idx > 0:
                grad = grad @ self.weights[layer_idx].T
                previous_z = pre_activations[layer_idx - 1]
                grad = grad * relu_derivative(previous_z)

        return {
            "weights": grad_weights,
            "biases": grad_biases,
        }

    def update_params(self, grads: Gradients, learning_rate: float) -> None:
        """
        Update model parameters using vanilla gradient descent.

        Args:
            grads:
                Gradients returned by backward().

            learning_rate:
                Step size for gradient descent.
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        grad_weights = grads["weights"]
        grad_biases = grads["biases"]

        if len(grad_weights) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} weight gradients, "
                f"got {len(grad_weights)}"
            )

        if len(grad_biases) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} bias gradients, " f"got {len(grad_biases)}"
            )

        for layer_idx in range(self.num_layers):
            if grad_weights[layer_idx].shape != self.weights[layer_idx].shape:
                raise ValueError(
                    f"Weight gradient shape mismatch at layer {layer_idx}: "
                    f"expected {self.weights[layer_idx].shape}, "
                    f"got {grad_weights[layer_idx].shape}"
                )

            if grad_biases[layer_idx].shape != self.biases[layer_idx].shape:
                raise ValueError(
                    f"Bias gradient shape mismatch at layer {layer_idx}: "
                    f"expected {self.biases[layer_idx].shape}, "
                    f"got {grad_biases[layer_idx].shape}"
                )

            self.weights[layer_idx] -= learning_rate * grad_weights[layer_idx]
            self.biases[layer_idx] -= learning_rate * grad_biases[layer_idx]

    def predict_logits(self, X: Array) -> Array:
        """
        Return raw logits for an input batch.
        """
        return self.forward(X, apply_softmax=False)

    def predict_proba(self, X: Array) -> Array:
        """
        Return class probabilities for an input batch.
        """
        return self.forward(X, apply_softmax=True)

    def predict(self, X: Array) -> Array:
        """
        Predict class labels for an input batch.
        """
        logits = self.predict_logits(X)
        return np.argmax(logits, axis=1)

    def parameters(self) -> list[Array]:
        """
        Return all trainable parameters.

        This is mainly useful for inspection and debugging.
        """
        params: list[Array] = []

        for weight, bias in zip(self.weights, self.biases):
            params.append(weight)
            params.append(bias)

        return params

    def num_parameters(self) -> int:
        """
        Return the total number of trainable scalar parameters.
        """
        return int(sum(param.size for param in self.parameters()))

    def summary(self) -> str:
        """
        Return a human-readable architecture summary.
        """
        lines = ["NeuralNet architecture:"]

        for layer_idx, (in_dim, out_dim) in enumerate(
            zip(self.layer_dims[:-1], self.layer_dims[1:])
        ):
            if layer_idx == self.num_layers - 1:
                activation = "linear logits"
            else:
                activation = "ReLU"

            lines.append(f"  Layer {layer_idx}: {in_dim} -> {out_dim} ({activation})")

        lines.append(f"Total parameters: {self.num_parameters()}")

        return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    model = NeuralNet(rng=rng)

    print(model.summary())

    X = rng.normal(size=(4, 784)).astype(np.float32)

    logits = model.forward(X)
    print("\nForward pass successful.")
    print(f"logits shape: {logits.shape}")

    dummy_dlogits = rng.normal(size=(4, 10)).astype(np.float32)

    grads = model.backward(dummy_dlogits)
    print("\nBackward pass successful.")

    for layer_idx, grad_weight in enumerate(grads["weights"]):
        print(f"dW{layer_idx} shape: {grad_weight.shape}")

    model.update_params(grads, learning_rate=0.01)
    print("\nParameter update successful.")

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    print(f"\nprobabilities shape: {probabilities.shape}")
    print(f"predictions shape: {predictions.shape}")
    print(f"predictions: {predictions}")
