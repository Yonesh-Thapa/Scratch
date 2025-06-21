import numpy as np

class TemporalContext:
    """Simple RNN for temporal/contextual processing."""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Wx = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b = np.zeros(hidden_dim)
        self.h = np.zeros(hidden_dim)

    def reset(self):
        self.h = np.zeros(self.hidden_dim)

    def step(self, x):
        self.h = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, self.h) + self.b)
        return self.h
