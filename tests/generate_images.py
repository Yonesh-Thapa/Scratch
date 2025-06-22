"""
Script to generate 28x28 grayscale images for all symbols (0-9, A-Z, a-z, and common symbols) using PIL.
Saves images to data/images/ as <symbol>.png
"""
import os
from PIL import Image, ImageDraw, ImageFont
import string

# Output directory
os.makedirs('data/images', exist_ok=True)

# Character set
ALL_SYMBOLS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase + "!@#$%^&*()-_=+[]{};:'\",.<>/?|\\`~")

# Try to use a monospaced font, fallback to default
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for symbol in ALL_SYMBOLS:
    img = Image.new('L', (28, 28), color=255)
    d = ImageDraw.Draw(img)
    # Use textbbox for accurate size (Pillow >=8.0), fallback to getsize
    try:
        bbox = d.textbbox((0, 0), symbol, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = font.getsize(symbol)
    # Center the symbol
    d.text(((28-w)//2, (28-h)//2), symbol, fill=0, font=font)
    # Save with safe filename
    safe_symbol = symbol if symbol.isalnum() else f'sym_{ord(symbol)}'
    img.save(f'data/images/{safe_symbol}.png')
print("All images generated in data/images/")
