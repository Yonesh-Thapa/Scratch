import numpy as np

class Recognizer:
    """
    Robust nearest-prototype classifier for SDR/latent vectors.
    Stores a prototype vector for each class/letter and recognizes new inputs by nearest neighbor search.
    """
    def __init__(self, letters, sdr_dim):
        self.letters = letters
        self.sdr_dim = sdr_dim
        self.prototypes = {l: np.zeros(sdr_dim) for l in letters}
        self.counts = {l: 0 for l in letters}

    def update(self, letter, sdr):
        """
        Update the prototype for a letter using running mean.
        Args:
            letter: class label
            sdr: SDR/latent vector (1D array)
        """
        if self.counts[letter] == 0:
            self.prototypes[letter] = sdr.copy()
        else:
            self.prototypes[letter] = (
                self.prototypes[letter] * self.counts[letter] + sdr
            ) / (self.counts[letter] + 1)
        self.counts[letter] += 1

    def recognize(self, sdr):
        """
        Recognize the class of a given SDR/latent vector by nearest prototype.
        Args:
            sdr: SDR/latent vector (1D array)
        Returns:
            label: recognized class label
            min_dist: distance to closest prototype
        """
        min_dist = float('inf')
        label = None
        for l, proto in self.prototypes.items():
            dist = np.linalg.norm(sdr - proto)
            if dist < min_dist:
                min_dist = dist
                label = l
        return label, min_dist
