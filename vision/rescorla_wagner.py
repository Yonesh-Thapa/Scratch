"""
Rescorla-Wagner learning rule for cue-outcome mapping in VisionModule.
"""
import numpy as np

class RescorlaWagnerLearner:
    def __init__(self, n_cues, n_outcomes, lr=0.01):
        self.V = np.zeros((n_cues, n_outcomes))
        self.lr = lr

    def predict(self, cues):
        return np.sum(self.V[cues, :], axis=0)

    def update(self, cues, outcome):
        pred = self.predict(cues)
        error = outcome - pred
        for cue in cues:
            self.V[cue, :] += self.lr * error
        return error
