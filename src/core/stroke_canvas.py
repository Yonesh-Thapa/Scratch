"""
stroke_canvas.py
Provides a canvas for drawing and extracting images.
"""
import numpy as np
import pygame

class StrokeCanvas:
    """Canvas for drawing and extracting images using pygame surfaces."""
    def __init__(self, width=28, height=28):
        self.width = width
        self.height = height
        pygame.init()
        self.surface = pygame.Surface((width, height))
        self.clear()

    def clear(self):
        self.surface.fill((0, 0, 0))

    def draw_stroke(self, points, color=(255, 255, 255), width=2):
        """
        Draws a stroke on the canvas.
        Args:
            points (list of (x, y)): List of points for the stroke.
            color (tuple): RGB color.
            width (int): Stroke width.
        """
        if len(points) > 1:
            pygame.draw.lines(self.surface, color, False, points, width)

    def get_image(self) -> np.ndarray:
        """
        Returns the current canvas as a grayscale numpy array.
        Returns:
            np.ndarray: 2D array (H, W) of grayscale image.
        """
        arr = pygame.surfarray.array3d(self.surface)
        arr = arr.mean(axis=2)  # Convert to grayscale
        arr = arr.T  # Pygame uses (W, H), numpy expects (H, W)
        arr = arr / 255.0
        return arr
