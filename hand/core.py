import os
import atexit
import numpy as np
from symbols import SYMBOLS
from .drawing import DigitalCanvas, DigitalPen
from .utils import check_similarity

CANVAS_SIZE = (28, 28)
MEMORY_PATH = 'hand_weights.npy'

class HandModule:
    def __init__(self, lr=0.5):
        self.input_dim = np.prod(CANVAS_SIZE)
        self.output_dim = len(SYMBOLS)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.2
        self.lr = lr
        self.symbols = SYMBOLS
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def draw_symbol(self, symbol_idx, actions=None):
        canvas = DigitalCanvas(size=CANVAS_SIZE)
        pen = DigitalPen(canvas, start_pos=(CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2))
        if actions is not None:
            for act in actions:
                if act['type'] == 'pen_down':
                    pen.pen_down()
                elif act['type'] == 'pen_up':
                    pen.pen_up()
                elif act['type'] == 'move':
                    pen.move_to(act['x'], act['y'], value=1.0, thickness=1)
        else:
            return self.W[symbol_idx].reshape(CANVAS_SIZE)
        return canvas.get_image()

    def update(self, symbol_idx, feedback_img, logger=None):
        flat_feedback = feedback_img.flatten()
        flat_feedback = (flat_feedback - np.mean(flat_feedback)) / (np.std(flat_feedback) + 1e-8)
        pred = self.W[symbol_idx]
        error = flat_feedback - pred
        if logger:
            logger.info(f"[HandModule] symbol_idx={symbol_idx} pred_mean={np.mean(pred):.4f} feedback_mean={np.mean(flat_feedback):.4f} error_mean={np.mean(error):.4f} error_std={np.std(error):.4f}")
        self.W[symbol_idx] += self.lr * error
        self.W[symbol_idx] = np.clip(self.W[symbol_idx], -10, 10)
        if logger:
            logger.info(f"[HandModule] Updated W[{symbol_idx}] mean={np.mean(self.W[symbol_idx]):.4f} std={np.std(self.W[symbol_idx]):.4f}")
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)
