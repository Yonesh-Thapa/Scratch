"""
letter_image_generator.py
Generates 28x28 grayscale images for A-Z and a-z using pygame fonts.
"""
import numpy as np
import pygame
import random

class LetterImageGenerator:
    """Generates 28x28 grayscale images for A-Z and a-z using pygame font rendering, with font and augmentation support."""
    def __init__(self, width=28, height=28, font_size=24, fonts=None):
        self.width = width
        self.height = height
        pygame.init()
        if fonts is None:
            fonts = ['arial', 'timesnewroman', 'comicsansms', 'calibri', 'couriernew', 'verdana']
        self.fonts = fonts
        self.font_size = font_size
        self.letters = [str(i) for i in range(10)]  # Only digits 0-9
        # Per-letter augmentation parameters
        self.default_aug_params = {
            'angle': 20,
            'tx': 2,
            'ty': 2,
            'scale_min': 0.9,
            'scale_max': 1.1,
            'noise': 0.05
        }
        self.aug_params = {letter: self.default_aug_params.copy() for letter in self.letters}

    def get_image(self, letter, font=None, augment=True, return_action=False):
        if font is None:
            font = random.choice(self.fonts)
        font_obj = pygame.font.SysFont(font, self.font_size, bold=True)
        surface = pygame.Surface((self.width, self.height))
        surface.fill((0, 0, 0))
        text = font_obj.render(letter, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width//2, self.height//2))
        surface.blit(text, text_rect)
        arr = pygame.surfarray.array3d(surface)
        arr = arr.mean(axis=2)  # Convert to grayscale
        arr = arr.T  # Pygame uses (W, H), numpy expects (H, W)
        arr = arr / 255.0
        action = {'font': font}
        if augment:
            params = self.aug_params.get(letter, self.default_aug_params)
            angle = random.uniform(-params['angle'], params['angle'])
            tx = random.randint(-params['tx'], params['tx'])
            ty = random.randint(-params['ty'], params['ty'])
            scale = random.uniform(params['scale_min'], params['scale_max'])
            arr = self.rotate(arr, angle)
            arr = self.translate(arr, tx, ty)
            arr = self.scale(arr, scale)
            arr += np.random.normal(0, params['noise'], arr.shape)
            arr = np.clip(arr, 0, 1)
            action.update({'angle': angle, 'tx': tx, 'ty': ty, 'scale': scale})
        if return_action:
            return arr, action
        return arr

    def increase_augmentation(self, letter):
        """Temporarily increase augmentation for a letter (for forced exploration/curriculum shift)."""
        # Increase augmentation parameters for this letter
        params = self.aug_params.get(letter, self.default_aug_params.copy())
        params['angle'] = min(params['angle'] + 5, 45)
        params['tx'] = min(params['tx'] + 1, 5)
        params['ty'] = min(params['ty'] + 1, 5)
        params['scale_min'] = max(params['scale_min'] - 0.02, 0.7)
        params['scale_max'] = min(params['scale_max'] + 0.02, 1.3)
        params['noise'] = min(params['noise'] + 0.01, 0.15)
        self.aug_params[letter] = params
        print(f"[LetterImageGenerator] Increased augmentation for letter {letter} (forced exploration)")

    def reduce_augmentation(self, letter):
        """Aggressively reduce augmentation for a letter (reset to minimum distortion/noise for recovery)."""
        min_params = {
            'angle': 0,
            'tx': 0,
            'ty': 0,
            'scale_min': 1.0,
            'scale_max': 1.0,
            'noise': 0.0
        }
        self.aug_params[letter] = min_params
        print(f"[LetterImageGenerator] Aggressively reduced augmentation for letter {letter} (full recovery mode)")

    def set_minimal_augmentation(self, letter):
        """Set augmentation to minimal for curriculum learning/debugging."""
        self.aug_params[letter] = {
            'angle': 0,
            'tx': 0,
            'ty': 0,
            'scale_min': 1.0,
            'scale_max': 1.0,
            'noise': 0.0
        }

    def set_medium_augmentation(self, letter):
        """Set augmentation to intermediate (medium) level for curriculum learning."""
        self.aug_params[letter] = {
            'angle': 10,
            'tx': 1,
            'ty': 1,
            'scale_min': 0.95,
            'scale_max': 1.05,
            'noise': 0.02
        }

    def augment(self, arr):
        # Random rotation
        angle = random.uniform(-20, 20)
        arr = self.rotate(arr, angle)
        # Random translation
        tx = random.randint(-2, 2)
        ty = random.randint(-2, 2)
        arr = self.translate(arr, tx, ty)
        # Random scaling
        scale = random.uniform(0.9, 1.1)
        arr = self.scale(arr, scale)
        # Add noise
        arr += np.random.normal(0, 0.05, arr.shape)
        arr = np.clip(arr, 0, 1)
        return arr

    def rotate(self, arr, angle):
        from scipy.ndimage import rotate
        return rotate(arr, angle, reshape=False, order=1, mode='nearest')

    def translate(self, arr, tx, ty):
        from scipy.ndimage import shift
        return shift(arr, [tx, ty], order=1, mode='nearest')

    def scale(self, arr, scale):
        from scipy.ndimage import zoom
        h, w = arr.shape
        arr_zoom = zoom(arr, scale, order=1)
        zh, zw = arr_zoom.shape
        # Center crop or pad
        out = np.zeros((h, w))
        y0 = max((zh - h) // 2, 0)
        x0 = max((zw - w) // 2, 0)
        y1 = min(h, zh)
        x1 = min(w, zw)
        out[:y1, :x1] = arr_zoom[y0:y0+y1, x0:x0+x1]
        return out

    def all_letter_images(self):
        return [(letter, self.get_image(letter, augment=False)) for letter in self.letters]
