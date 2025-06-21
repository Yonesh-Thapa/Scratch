"""
factorial_encoder.py
Encodes features into a sparse, high-dimensional vector (SDR) using winner-take-all.
"""
import numpy as np
from .probabilistic_encoder import ProbabilisticEncoder

class FactorialEncoder:
    """Encodes features into a sparse distributed representation (SDR) using winner-take-all and probabilistic encoding."""
    def __init__(self, input_dim, sdr_dim, sparsity=0.05, seed=42, latent_dim=64):
        self.input_dim = input_dim
        self.sdr_dim = sdr_dim
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)
        # Random projection matrix
        self.proj = self.rng.standard_normal((input_dim, sdr_dim))
        self.prob_encoder = ProbabilisticEncoder(input_dim, latent_dim)

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encodes features into a sparse high-dimensional vector (SDR).
        Args:
            features (np.ndarray): 1D array of features.
        Returns:
            np.ndarray: 1D binary SDR vector.
        """
        # Probabilistic encoding
        z, mu, logvar = self.prob_encoder.encode(features)
        x = features @ self.proj  # Linear projection
        k = int(self.sdr_dim * self.sparsity)
        idx = np.argpartition(x, -k)[-k:]
        sdr = np.zeros(self.sdr_dim, dtype=np.float32)
        sdr[idx] = 1.0
        return sdr, z, mu, logvar

    def grow(self, n_new):
        """Add n_new neurons (columns) to the projection matrix."""
        new_proj = self.rng.standard_normal((self.input_dim, n_new))
        self.proj = np.concatenate([self.proj, new_proj], axis=1)
        self.sdr_dim += n_new

    def prune(self, inactive_indices):
        """Remove neurons (columns) from the projection matrix by index."""
        self.proj = np.delete(self.proj, inactive_indices, axis=1)
        self.sdr_dim = self.proj.shape[1]

    def rewire(self, indices):
        """Randomize the projection for given neuron indices."""
        for idx in indices:
            self.proj[:, idx] = self.rng.standard_normal(self.input_dim)
