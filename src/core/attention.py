import numpy as np

class Attention:
    """Spatial attention mechanism for 2D images."""
    def __init__(self, input_shape, num_heads=1):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.W = np.random.randn(num_heads, *input_shape) * 0.01
        self.b = np.zeros((num_heads, 1, 1))

    def get_attention_map(self, x):
        # x: (H, W)
        attn_scores = np.zeros((self.num_heads, *self.input_shape))
        for h in range(self.num_heads):
            attn_scores[h] = self.W[h] * x + self.b[h]
        attn_map = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=(1,2), keepdims=True)
        return attn_map

    def apply_attention(self, x):
        attn_map = self.get_attention_map(x)
        attended = np.sum(attn_map * x, axis=(1,2))
        return attended, attn_map
