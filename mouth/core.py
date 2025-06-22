import os
import atexit
import numpy as np
from symbols import SYMBOLS
from .articulator import DigitalArticulator
from .utils import xavier_init, LRScheduler

MEMORY_PATH = 'mouth_weights.npy'

class MouthModule:
    def __init__(self, lr=0.01, wave_dim=8000, sr=16000):
        self.input_dim = wave_dim
        self.output_dim = len(SYMBOLS)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        self.articulator = DigitalArticulator(wave_dim=wave_dim, sr=sr)
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def synthesize(self, symbol_idx):
        # For now, just return the weights as the waveform
        return self.W[symbol_idx]

    def update(self, symbol_idx, feedback_wave, logger=None):
        flat_feedback = feedback_wave.flatten()
        flat_feedback = (flat_feedback - np.mean(flat_feedback)) / (np.std(flat_feedback) + 1e-8)
        pred = self.W[symbol_idx]
        error = flat_feedback - pred
        if logger:
            logger.info(f"[MouthModule] symbol_idx={symbol_idx} pred_mean={np.mean(pred):.4f} feedback_mean={np.mean(flat_feedback):.4f} error_mean={np.mean(error):.4f} error_std={np.std(error):.4f}")
        self.W[symbol_idx] += self.lr * error
        self.W[symbol_idx] = np.clip(self.W[symbol_idx], -10, 10)
        if logger:
            logger.info(f"[MouthModule] Updated W[{symbol_idx}] mean={np.mean(self.W[symbol_idx]):.4f} std={np.std(self.W[symbol_idx]):.4f}")
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)
