import numpy as np
import logging

# Xavier/He initialization utility
def xavier_init(shape):
    fan_in, fan_out = shape[1], shape[0]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

class LRScheduler:
    def __init__(self, lr, factor=0.5, patience=5, min_lr=1e-5):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self.wait = 0
        return self.lr

def check_similarity(img1, img2):
    img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return -np.mean((img1.flatten() - img2.flatten())**2)
