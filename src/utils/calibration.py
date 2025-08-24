import numpy as np

class TemperatureScaler:
    def __init__(self):
        self.t = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, iters: int = 200):
        # Simple 1D optimization per-class merged; for true per-class, fit per column
        for _ in range(iters):
            probs = 1 / (1 + np.exp(-logits / self.t))
            grad = np.mean((probs - labels) * logits) / (self.t**2 + 1e-8)
            self.t -= lr * grad
            self.t = max(self.t, 1e-3)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-logits / self.t))
