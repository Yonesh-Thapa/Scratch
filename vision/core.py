import os
import atexit
import numpy as np
from symbols import SYMBOLS
from .retina import DigitalRetina
from .utils import xavier_init, LRScheduler

MEMORY_PATH = 'vision_weights.npy'

class VisionModule:
    def __init__(self, learning_rule='delta', lr=0.1, retina_fov=(28,28), retina_img_shape=(128,128)):
        if retina_fov[0] < retina_img_shape[0] or retina_fov[1] < retina_img_shape[1]:
            self.retina_fov = retina_img_shape
            self.fov_covers_whole_image = True
        else:
            self.retina_fov = retina_fov
            self.fov_covers_whole_image = False
        self.input_dim = self.retina_fov[0] * self.retina_fov[1]
        self.output_dim = len(SYMBOLS)
        # Import learning rule dynamically
        if learning_rule == 'delta':
            from .delta_rule import DeltaRuleLearner
            self.learner = DeltaRuleLearner(self.input_dim, self.output_dim, lr)
        else:
            from .rescorla_wagner import RescorlaWagnerLearner
            self.learner = RescorlaWagnerLearner(self.input_dim, self.output_dim, lr)
        self.symbols = SYMBOLS
        self.retina = DigitalRetina(fov_shape=self.retina_fov, image_shape=retina_img_shape)
        if os.path.exists(MEMORY_PATH):
            try:
                W = np.load(MEMORY_PATH)
                if W.shape == (self.output_dim, self.input_dim):
                    self.learner.W = W
                else:
                    pass  # shape mismatch, skip
            except Exception:
                pass
        atexit.register(self.save_memory)

    def save_memory(self):
        np.save(MEMORY_PATH, self.learner.W)

    def recognize(self, image):
        img_shape = image.shape
        x, y = self._random_retina_position(img_shape)
        self.retina.set_position(x, y)
        fov = self.retina.get_view(image)
        x_flat = fov.astype(np.float32).flatten()
        y_pred = self.learner.predict(x_flat)
        idx = np.argmax(y_pred)
        symbol = self.symbols[idx]
        return symbol, y_pred, (x, y)

    def learn(self, image, target_idx):
        img_shape = image.shape
        x, y = self._random_retina_position(img_shape)
        self.retina.set_position(x, y)
        fov = self.retina.get_view(image)
        x_flat = fov.flatten()
        t = np.zeros(self.output_dim)
        t[target_idx] = 1.0
        pred = self.learner.predict(x_flat)
        error = self.learner.update(x_flat, t)
        return error

    def _random_retina_position(self, img_shape):
        if self.fov_covers_whole_image:
            return 0, 0
        h, w = img_shape[:2]
        fh, fw = self.retina_fov
        max_x = max(0, h - fh)
        max_y = max(0, w - fw)
        x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        return x, y
