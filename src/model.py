import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ReLU activation"""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last dimension.
    x: (batch_size, num_classes)
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNet:
    """
    Fully connected MLP for MNIST: 784 -> 128 -> 64 -> 10.
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

        # Parameters
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            W = weight_scale * self.rng.standard_normal(
                size=(in_dim, out_dim), dtype=np.float32
            )
            b = np.zeros((1, out_dim), dtype=np.float32)

            self.weights.append(W)
            self.biases.append(b)

        self._cache: dict[str, list[np.ndarray]] = {}

    def forward(self, X: np.ndarray, apply_softmax: bool = False) -> np.ndarray:
        """
        Forward pass through the network

        X: (batch_size, input_dim)
        returns:
            logits if apply_softmax = False
            probabilities if apply_softmax=True
        """
        activations = [X]
        pre_activations = []

        A = X
        for layer_idx in range(self.num_layers - 1):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]

            Z = A @ W + b
            A = relu(Z)

            pre_activations.append(Z)
            activations.append(A)

        # Output layer: linear only (no activation here)
        W_out = self.weights[-1]
        b_out = self.biases[-1]
        logits = activations[-1] @ W_out + b_out  # (batch_size, output_dim)

        # Cache for backward
        self._cache["activations"] = activations
        self._cache["pre_activations"] = pre_activations
        self._cache["logits"] = logits

        if apply_softmax:
            return softmax(logits)
        return logits

    def backward(
        self,
        dlogits: np.ndarray,
        cache: dict[str, list[np.ndarray] | np.ndarray],
    ) -> dict[str, list[np.ndarray]]:
        """
        Run backpropagation through the network.

        Args:
            dlogits:
                Gradient of the loss with respect to the output logits
                Shape: (batch_size, output_size)

            cache:
                Cache returned by the forward pass
        Returns:
            A dictionary containing gradients for weights and biases
        """
        activations = cache["activations"]
        pre_activations = cache["pre-activations"]

        if not isinstance(activations, list):
            raise TypeError("cache['activations'] must be a list")

        if not isinstance(pre_activations, list):
            raise TypeError("cache['pre_activations'] must be a list")

        grad_weights: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grad_biases: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        grad = dlogits

        for layer_index in reversed(range(len(self.weights))):
            previous_activation = activations[layer_index]

            grad_weights[layer_index] = previous_activation.T @ grad
            grad_biases[layer_index] = np.sum(grad, axis=0, keepdims=True)

            if layer_index > 0:
                grad = grad @ self.weights[layer_index].T
                previous_z = pre_activations[layer_index - 1]
                grad = grad * self.relu_derivative(previous_z)

        return {
            "weights": grad_weights,
            "biases": grad_biases,
        }

    def update_params(
        self,
        grads: dict[str, list[np.ndarray]],
        learning_rate: float,
    ) -> None:
        """
        Update model parameters using gradient descent

        Args:
            grads:
                Gradients returned by backward

            learning_rate:
                Step size for gradient descent
        """

        grad_weights = grads["weights"]
        grad_biases = grads["biases"]

        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= learning_rate * grad_weights[layer_index]
            self.biases[layer_index] -= learning_rate * grad_biases[layer_index]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict class labels for an input batch
        """
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)


if __name__ == "__main__":
    # quick sanity test
    model = NeuralNet()

    x = np.random.randn(4, 784).astype(np.float32)
    logits, cache = model.forward(x)

    print("Forward pass successful")
    print(f"logits shape: {logits.shape}")

    dummy_dlogits = np.random.randn(4, 10).astype(np.float32)
    grads = model.backward(dummy_dlogits, cache)
    model.update_params(grads, learning_rate=0.01)

    print("Backward pass and parameter update successful")
    print(f"number of weights matrices : {len(grads['weights'])}")
