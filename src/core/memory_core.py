"""
memory_core.py
Handles saving and loading of learned decoder weights only.
"""
import numpy as np
import os

class MemoryCore:
    """Saves and loads only the learned decoder weights (no raw data)."""
    def __init__(self, path='decoder_weights.npy'):
        self.path = path

    def save(self, weights):
        np.save(self.path, weights)
        print(f"[MemoryCore] Saved decoder weights to {self.path}")

    def load(self):
        if os.path.exists(self.path):
            weights = np.load(self.path)
            print(f"[MemoryCore] Loaded decoder weights from {self.path}")
            return weights
        else:
            print(f"[MemoryCore] No saved weights found at {self.path}")
            return None
