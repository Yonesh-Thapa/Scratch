"""
EarModule: Listens to real audio samples, extracts features, learns audio-symbol associations.
Runs as an independent process, communicates via sockets/IPC.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import os
import numpy as np
import socket
import pickle
import librosa
from symbols import SYMBOLS
import atexit
import signal
import sys

try:
    from ear.ear_module import EarModule
except ImportError:
    pass  # No internal imports needed, placeholder for consistency

# Config
HOST = '127.0.0.1'
PORT = 5003
SAMPLE_RATE = 16000
N_MFCC = 13
MEMORY_PATH = 'ear_weights.npy'

class EarModule:
    def __init__(self, lr=0.01):
        self.input_dim = N_MFCC
        self.output_dim = len(SYMBOLS)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def recognize(self, x):
        y = np.dot(self.W, x)
        idx = np.argmax(y)
        return self.symbols[idx], y

    def update(self, x, target_idx):
        target = np.zeros(self.output_dim)
        target[target_idx] = 1.0
        pred = np.dot(self.W, x)
        error = target - pred
        self.W += self.lr * np.outer(error, x)
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[EarModule] Listening on {HOST}:{PORT}")
            while True:
                conn, addr = s.accept()
                with conn:
                    data = b''
                    while True:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet
                    msg = pickle.loads(data)
                    if msg['cmd'] == 'recognize':
                        x = msg['data']
                        label, y = self.recognize(x)
                        conn.sendall(pickle.dumps({'label': label, 'y': y}))
                    elif msg['cmd'] == 'learn':
                        x = msg['data']
                        target_idx = msg['target_idx']
                        error = self.update(x, target_idx)
                        conn.sendall(pickle.dumps({'error': error}))

def signal_handler(sig, frame):
    print('Exiting EarModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    em = EarModule(lr=args.lr)
    em.run()
