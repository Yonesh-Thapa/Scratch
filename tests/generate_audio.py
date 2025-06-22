"""
Script to generate simple synthetic audio (sine wave) for all symbols (0-9, A-Z, a-z, and common symbols).
Saves WAV files to data/audio/ as <symbol>.wav
"""
import os
import numpy as np
import soundfile as sf
import string

os.makedirs('data/audio', exist_ok=True)

ALL_SYMBOLS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase + "!@#$%^&*()-_=+[]{};:'\",.<>/?|\\`~")
SAMPLE_RATE = 16000
DURATION = 0.5  # seconds

# Assign a unique frequency to each symbol
base_freq = 300
freq_step = 15
for idx, symbol in enumerate(ALL_SYMBOLS):
    freq = base_freq + idx * freq_step
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    safe_symbol = symbol if symbol.isalnum() else f'sym_{ord(symbol)}'
    sf.write(f'data/audio/{safe_symbol}.wav', audio, SAMPLE_RATE)
print("All audio files generated in data/audio/")
