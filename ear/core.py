import os
import atexit
import numpy as np
from symbols import SYMBOLS
from .cochlea import DigitalCochlea
from .utils import xavier_init, LRScheduler

MEMORY_PATH = 'ear_weights.npy'

class EarModule:
    def __init__(self, lr=0.01, cochlea_filters=13, sr=16000):
        self.input_dim = cochlea_filters
        self.output_dim = len(SYMBOLS)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        self.cochlea = DigitalCochlea(n_filters=cochlea_filters, sr=sr)
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def recognize(self, waveform):
        feat = self.cochlea.process(waveform)
        feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
        y = np.dot(self.W, feat)
        idx = np.argmax(y)
        return self.symbols[idx], y

    def update(self, idx, waveform, logger=None):
        feat = self.cochlea.process(waveform)
        feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
        target = np.zeros(self.output_dim)
        target[idx] = 1.0
        y = np.dot(self.W, feat)
        error = target - (y == y.max()).astype(np.float32)
        if logger:
            logger.info(f"[EarModule] idx={idx} pred={y} target={target} error={error}")
        self.W += self.lr * error[:, None] * feat[None, :]
        self.W = np.clip(self.W, -10, 10)
        if logger:
            logger.info(f"[EarModule] Updated W mean={np.mean(self.W):.4f} std={np.std(self.W):.4f}")
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)
