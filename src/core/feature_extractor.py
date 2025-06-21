"""
feature_extractor.py
Extracts edge features from images using Sobel filters.
"""
import numpy as np
from scipy.ndimage import convolve
from .attention import Attention

class FeatureExtractor:
    """Extracts edge features from grayscale images using Sobel filters and applies attention."""
    def __init__(self, input_shape=(28,28), num_heads=1):
        # Sobel kernels
        self.sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        self.sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        self.attention = Attention(input_shape, num_heads)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts edge features from a grayscale image.
        Args:
            image (np.ndarray): 2D array (H, W) of grayscale image.
        Returns:
            np.ndarray: 3D array (H, W, 2) with Sobel X and Y responses.
        """
        gx = convolve(image, self.sobel_x)
        gy = convolve(image, self.sobel_y)
        features = np.stack([gx, gy], axis=-1)
        # Apply attention to image, get attention map
        attn_map = self.attention.get_attention_map(image)
        # If multiple heads, average or concatenate as needed
        if attn_map.shape[0] == 1:
            attn_map = attn_map[0]
        # Expand dims to match features for concatenation
        attn_map_expanded = attn_map[..., np.newaxis]
        features = np.concatenate([features, attn_map_expanded], axis=-1)
        return features
