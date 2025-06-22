"""
HandModule: Learns to write/draw symbols on a digital canvas using Delta Rule/Rescorla-Wagner learning.
Runs as an independent process, communicates via sockets/IPC.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import os
import numpy as np
import socket
import pickle
import pygame
from PIL import Image
from symbols import SYMBOLS
import atexit
import signal
import sys

# Attempt to import HandModule from hand.hand_module, if not available, continue without it
try:
    from hand.hand_module import HandModule
except ImportError:
    pass  # No internal imports needed, placeholder for consistency

# Config
HOST = '127.0.0.1'
PORT = 5002
INPUT_SHAPE = (28, 28)
CANVAS_SIZE = (28, 28)
MEMORY_PATH = 'hand_weights.npy'

class HandModule:
    def __init__(self, lr=0.01):
        self.input_dim = len(SYMBOLS)
        self.output_dim = np.prod(CANVAS_SIZE)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        pygame.init()
        self.screen = pygame.Surface(CANVAS_SIZE)
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def draw_symbol(self, symbol_idx):
        if symbol_idx < 0 or symbol_idx >= len(self.symbols):
            print(f"[HandModule] Invalid symbol_idx: {symbol_idx}")
            return np.zeros(CANVAS_SIZE, dtype=np.float32)
        x = np.zeros(self.input_dim)
        x[symbol_idx] = 1.0
        img_flat = np.dot(self.W, x)
        img = img_flat.reshape(CANVAS_SIZE)
        img = np.clip(img, 0, 1)
        return img
    
    def update(self, symbol_idx, feedback_img):
        if symbol_idx < 0 or symbol_idx >= len(self.symbols):
            print(f"[HandModule] Invalid symbol_idx for update: {symbol_idx}")
            return np.zeros(np.prod(CANVAS_SIZE), dtype=np.float32)
        x = np.zeros(self.input_dim)
        x[symbol_idx] = 1.0
        pred_img_flat = np.dot(self.W, x)
        error = feedback_img.flatten() - pred_img_flat
        self.W += self.lr * np.outer(error, x)
        return error

    

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)

    def run(self):
        with conn:
            try:
                data = b''
                while True:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet
                msg = pickle.loads(data)
                if msg['cmd'] == 'draw':
                    symbol_idx = msg['symbol_idx']
                    img = self.draw_symbol(symbol_idx)
                    conn.sendall(pickle.dumps({'img': img}))
                elif msg['cmd'] == 'learn':
                    symbol_idx = msg['symbol_idx']
                    feedback_img = msg['feedback_img']
                    error = self.update(symbol_idx, feedback_img)
                    conn.sendall(pickle.dumps({'error': error}))
            except Exception as e:
                print(f"[HandModule] Exception: {e}")
                conn.sendall(pickle.dumps({'error': str(e)}))
def signal_handler(sig, frame):
    print('Exiting HandModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    hm = HandModule(lr=args.lr)
    hm.run()
