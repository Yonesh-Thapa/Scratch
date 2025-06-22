import numpy as np

class DigitalCanvas:
    """
    Simulates a digital canvas for drawing with a pen.
    """
    def __init__(self, size=(28, 28)):
        self.size = size
        self.clear()

    def clear(self):
        self.img = np.zeros(self.size, dtype=np.float32)

    def draw_line(self, x0, y0, x1, y1, value=1.0, thickness=1):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self._draw_point(x, y, value, thickness)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                self._draw_point(x, y, value, thickness)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self._draw_point(x, y, value, thickness)

    def _draw_point(self, x, y, value, thickness):
        for dx in range(-thickness, thickness+1):
            for dy in range(-thickness, thickness+1):
                xx, yy = x+dx, y+dy
                if 0 <= xx < self.size[0] and 0 <= yy < self.size[1]:
                    self.img[xx, yy] = value

    def get_image(self):
        return np.clip(self.img, 0, 1)

class DigitalPen:
    """
    Simulates a digital pen that moves and draws on a canvas.
    """
    def __init__(self, canvas, start_pos=(0,0)):
        self.canvas = canvas
        self.x, self.y = start_pos
        self.down = False

    def pen_down(self):
        self.down = True

    def pen_up(self):
        self.down = False

    def move_to(self, x, y, value=1.0, thickness=1):
        if self.down:
            self.canvas.draw_line(self.x, self.y, x, y, value, thickness)
        self.x, self.y = x, y
