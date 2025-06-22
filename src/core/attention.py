import numpy as np

class Attention:
    """Spatial attention mechanism for 2D images with learnable weights and basic update step."""
    def __init__(self, input_shape, num_heads=4, init_scale=2.0, temperature=2.0):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.W = np.random.randn(num_heads, *input_shape) * init_scale
        self.b = np.zeros((num_heads, 1, 1))
        self.lr = 0.05  # Higher LR for attention
        self.temperature = temperature

    def get_attention_map(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        W = np.nan_to_num(self.W, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(self.b, nan=0.0, posinf=0.0, neginf=0.0)
        attn_scores = np.zeros((self.num_heads, *self.input_shape))
        for h in range(self.num_heads):
            attn_scores[h] = W[h] * x + b[h]
        attn_scores = attn_scores - np.max(attn_scores, axis=(1,2), keepdims=True)
        attn_scores = np.clip(attn_scores, -10, 10)
        exp_scores = np.exp(attn_scores / self.temperature)
        denom = np.sum(exp_scores, axis=(1,2), keepdims=True) + 1e-8
        attn_map = exp_scores / denom
        attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)
        return attn_map

    def apply_attention(self, x):
        attn_map = self.get_attention_map(x)  # shape: (num_heads, H, W)
        # Average over all heads to get a single attention map
        avg_attn_map = np.mean(attn_map, axis=0)  # shape: (H, W)
        attended = avg_attn_map * x               # elementwise
        attended = np.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        return attended, avg_attn_map

    def update(self, x, grad_attn_map):
        for h in range(self.num_heads):
            grad_W = grad_attn_map[h] * x
            self.W[h] -= self.lr * grad_W
