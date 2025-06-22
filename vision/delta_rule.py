"""
Delta Rule implementation for visual feature learning in VisionModule.
"""
import numpy as np

class DeltaRuleLearner:
    def __init__(self, input_dim, output_dim, lr=0.1):
        # Use higher learning rate for faster convergence
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.lr = lr

    def predict(self, x):
        # Ensure input is 1D and normalized
        x = np.asarray(x).flatten()
        return np.dot(self.W, x)

    def update(self, x, target):
        x = np.asarray(x).flatten()
        y = self.predict(x)
        error = target - y
        print(f"[DeltaRuleLearner] Before update: W[0, :5]={self.W[0, :5]}")
        self.W += self.lr * np.outer(error, x)
        # Clip weights to prevent explosion
        self.W = np.clip(self.W, -10, 10)
        print(f"[DeltaRuleLearner] After update: W[0, :5]={self.W[0, :5]}")
        print(f"[DeltaRuleLearner] Error: {error[:5]}")
        return error
