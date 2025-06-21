"""
letter_decoder.py
Decodes latent SDR vectors back to image space.
"""
import numpy as np
from .probabilistic_encoder import ProbabilisticEncoder

class LetterDecoder:
    """Decodes SDRs back to image space using a learned linear mapping and probabilistic decoding."""
    def __init__(self, sdr_dim, output_dim, seed=42, latent_dim=64):
        self.sdr_dim = sdr_dim
        self.output_dim = output_dim
        self.rng = np.random.default_rng(seed)
        # Linear weights for reconstruction
        self.W = self.rng.standard_normal((sdr_dim, output_dim)) * 0.1
        self.prob_decoder = ProbabilisticEncoder(sdr_dim, latent_dim)

    def decode(self, sdr: np.ndarray) -> np.ndarray:
        """
        Decodes an SDR vector back to image space.
        Args:
            sdr (np.ndarray): 1D binary SDR vector.
        Returns:
            np.ndarray: 1D reconstructed image vector.
        """
        # Probabilistic decoding
        z, mu, logvar = self.prob_decoder.encode(sdr)
        return sdr @ self.W

    def update(self, sdr: np.ndarray, target: np.ndarray, lr=0.01):
        """
        Updates the decoder weights using error-driven learning.
        Args:
            sdr (np.ndarray): 1D binary SDR vector.
            target (np.ndarray): 1D target image vector.
            lr (float): Learning rate.
        """
        pred = self.decode(sdr)
        error = target - pred
        self.W += lr * np.outer(sdr, error)

    def grow(self, n_new):
        """Add n_new neurons (rows) to the decoder weights."""
        new_W = self.rng.standard_normal((n_new, self.output_dim)) * 0.1
        self.W = np.concatenate([self.W, new_W], axis=0)
        self.sdr_dim += n_new

    def prune(self, inactive_indices):
        """Remove neurons (rows) from the decoder weights by index."""
        self.W = np.delete(self.W, inactive_indices, axis=0)
        self.sdr_dim = self.W.shape[0]
