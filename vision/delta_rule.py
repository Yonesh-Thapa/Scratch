"""
Delta Rule implementation for visual feature learning in VisionModule.
"""
import numpy as np

class DeltaRuleLearner:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.lr = lr

    def predict(self, x):
        return np.dot(self.W, x)

    def update(self, x, target):
        y = self.predict(x)
        error = target - y
        self.W += self.lr * np.outer(error, x)
        return error
