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


if __name__ == "__main__":
    # quick sanity test or demo
    BATCH_SIZE = 5
    X_fake = np.random.rand(BATCH_SIZE, 784).astype(np.float32)

    model = NeuralNet()
    logits = model.forward(X_fake)
    probs = model.forward(X_fake, apply_softmax=True)

    print("logits shape:", logits.shape)
    print("probs shape:", probs.shape)
    print("row sums:", probs.sum(axis=1))
