import numpy as np

class DigitalRetina:
    """
    Simulates a digital retina (field of view) that can move over a larger image.
    Handles centering/padding if image is smaller than FOV.
    """
    def __init__(self, fov_shape=(28, 28), image_shape=(128, 128)):
        self.fov_shape = fov_shape
        self.image_shape = image_shape
        self.x = 0  # top-left x of FOV
        self.y = 0  # top-left y of FOV

    def set_position(self, x, y):
        self.x = int(np.clip(x, 0, max(0, self.image_shape[0] - self.fov_shape[0])))
        self.y = int(np.clip(y, 0, max(0, self.image_shape[1] - self.fov_shape[1])))

    def get_view(self, image):
        h, w = image.shape
        fh, fw = self.fov_shape
        if h < fh or w < fw:
            pad_h = max(0, fh - h)
            pad_w = max(0, fw - w)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
            h, w = image.shape
        x, y = self.x, self.y
        x = int(np.clip(x, 0, h - fh))
        y = int(np.clip(y, 0, w - fw))
        return image[x:x+fh, y:y+fw]
