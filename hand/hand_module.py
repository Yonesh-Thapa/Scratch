"""
HandModule: Learns to draw symbol shapes, receives feedback, updates internal model.
Runs as an independent process, communicates via sockets/IPC.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import os
try:
    import numpy as np
except ImportError:
    print("[HandModule] numpy is required but not installed")
    sys.exit(1)
import socket
import pickle
import atexit
import signal
import sys
import struct
from symbols import SYMBOLS

# Config
HOST = '127.0.0.1'
PORT = 5002
CANVAS_SIZE = (28, 28)
MEMORY_PATH = 'hand_weights.npy'

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

class HandModule:
    def __init__(self, lr=0.01):
        self.input_dim = np.prod(CANVAS_SIZE)
        self.output_dim = len(SYMBOLS)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def draw_symbol(self, symbol_idx):
        return self.W[symbol_idx].reshape(CANVAS_SIZE)

    def update(self, symbol_idx, feedback_img):
        flat_feedback = feedback_img.flatten()
        # Normalize feedback
        flat_feedback = (flat_feedback - np.mean(flat_feedback)) / (np.std(flat_feedback) + 1e-8)
        pred = self.W[symbol_idx]
        error = flat_feedback - pred
        self.W[symbol_idx] += self.lr * error
        # Clip weights
        self.W[symbol_idx] = np.clip(self.W[symbol_idx], -10, 10)
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[HandModule] Listening on {HOST}:{PORT}")
            while True:
                try:
                    conn, addr = s.accept()
                    with conn:
                        header = recvall(conn, 4)
                        if not header:
                            continue
                        msg_len = struct.unpack('>I', header)[0]
                        data = recvall(conn, msg_len)
                        if not data:
                            continue
                        try:
                            msg = pickle.loads(data)
                            print(f"[HandModule] Decoded message: {msg}")
                        except Exception as e:
                            print(f"[HandModule] Error decoding message: {e}")
                            resp = pickle.dumps({'error': f'Error decoding message: {e}'})
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            continue
                        if msg['cmd'] == 'draw':
                            symbol_idx = msg['symbol_idx']
                            print(f"[HandModule] Processing draw command. symbol_idx: {symbol_idx}")
                            if 0 <= symbol_idx < len(self.symbols):
                                img = self.draw_symbol(symbol_idx)
                            else:
                                print(f"[HandModule] Invalid symbol_idx: {symbol_idx}")
                                img = np.zeros(CANVAS_SIZE, dtype=np.float32)
                            response = {'img': img}
                        elif msg['cmd'] == 'learn':
                            symbol_idx = msg['symbol_idx']
                            feedback_img = msg['feedback_img']
                            print(f"[HandModule] Processing learn command. symbol_idx: {symbol_idx}, feedback_img type: {type(feedback_img)}, shape: {getattr(feedback_img, 'shape', None)}")
                            if 0 <= symbol_idx < len(self.symbols):
                                error = self.update(symbol_idx, feedback_img)
                            else:
                                print(f"[HandModule] Invalid symbol_idx for learn: {symbol_idx}")
                                error = np.zeros(np.prod(CANVAS_SIZE))
                            response = {'error': error}
                        elif msg['cmd'] == 'shutdown':
                            print('[HandModule] Shutdown command received. Exiting...')
                            response = {'status': 'shutting down'}
                            resp = pickle.dumps(response)
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            sys.exit(0)
                        else:
                            response = {'error': 'Unknown command'}
                        resp = pickle.dumps(response)
                        resp_len = struct.pack('>I', len(resp))
                        conn.sendall(resp_len + resp)
                except Exception as e:
                    print(f"[HandModule] Connection error: {e}")

def signal_handler(sig, frame):
    print('Exiting HandModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__' or __name__.endswith('.hand_module'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    hm = HandModule(lr=args.lr)
    hm.run()
