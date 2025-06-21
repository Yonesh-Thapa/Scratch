import numpy as np

class Attention:
    """Spatial attention mechanism for 2D images with learnable weights and basic update step."""
    def __init__(self, input_shape, num_heads=1, init_scale=1.0):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.W = np.random.randn(num_heads, *input_shape) * init_scale  # Increased scale for more contrast
        self.b = np.zeros((num_heads, 1, 1))
        self.lr = 0.01  # Learning rate for attention weights

    def get_attention_map(self, x):
        # Robust: replace NaN/Inf in input and weights
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        W = np.nan_to_num(self.W, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(self.b, nan=0.0, posinf=0.0, neginf=0.0)
        attn_scores = np.zeros((self.num_heads, *self.input_shape))
        for h in range(self.num_heads):
            attn_scores[h] = W[h] * x + b[h]
        # Subtract max per head for numerical stability
        attn_scores = attn_scores - np.max(attn_scores, axis=(1,2), keepdims=True)
        # Clamp scores to avoid overflow in exp
        attn_scores = np.clip(attn_scores, -30, 30)
        exp_scores = np.exp(attn_scores)
        # Add epsilon to denominator to avoid division by zero
        denom = np.sum(exp_scores, axis=(1,2), keepdims=True) + 1e-8
        attn_map = exp_scores / denom
        # Replace any NaN/Inf in output with zeros
        attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)
        return attn_map

    def apply_attention(self, x):
        attn_map = self.get_attention_map(x)
        attended = np.sum(attn_map * x, axis=(1,2))
        # Robust: replace NaN/Inf in output
        attended = np.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        return attended, attn_map

    def update(self, x, grad_attn_map):
        # Simple gradient update for attention weights (optional, for learning)
        # grad_attn_map: gradient of loss w.r.t. attn_map, shape (num_heads, H, W)
        for h in range(self.num_heads):
            grad_W = grad_attn_map[h] * x  # Elementwise product
            self.W[h] -= self.lr * grad_W
