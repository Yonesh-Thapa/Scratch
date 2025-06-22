"""
EarModule: Learns to recognize audio features of spoken symbols.
Runs as an independent process, communicates via sockets/IPC.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import os
try:
    import numpy as np
except ImportError:
    print("[EarModule] numpy is required but not installed")
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
PORT = 5003
MEMORY_PATH = 'ear_weights.npy'
FEAT_DIM = 13  # Typically MFCC dimension

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

class EarModule:
    def __init__(self, lr=0.01):
        self.input_dim = FEAT_DIM
        self.output_dim = len(SYMBOLS)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def recognize(self, feat):
        y = np.dot(self.W, feat)
        idx = np.argmax(y)
        return self.symbols[idx], y

    def update(self, idx, feat):
        target = np.zeros(self.output_dim)
        target[idx] = 1.0
        y = np.dot(self.W, feat)
        error = target - (y == y.max()).astype(np.float32)
        self.W += self.lr * error[:, None] * feat[None, :]
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
                            print(f"[EarModule] Decoded message: {msg}")
                        except Exception as e:
                            print(f"[EarModule] Error decoding message: {e}")
                            resp = pickle.dumps({'error': f'Error decoding message: {e}'})
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            continue
                        if msg['cmd'] == 'recognize':
                            feat = msg['data']
                            print(f"[EarModule] Processing recognize command. Data type: {type(feat)}, shape: {getattr(feat, 'shape', None)}")
                            label, y = self.recognize(feat)
                            response = {'label': label, 'y': y}
                        elif msg['cmd'] == 'learn':
                            feat = msg['data']
                            target_idx = msg['target_idx']
                            print(f"[EarModule] Processing learn command. Data type: {type(feat)}, shape: {getattr(feat, 'shape', None)}, target_idx: {target_idx}")
                            error = self.update(target_idx, feat)
                            response = {'error': error}
                        elif msg['cmd'] == 'shutdown':
                            print('[EarModule] Shutdown command received. Exiting...')
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
                    print(f"[EarModule] Connection error: {e}")

def signal_handler(sig, frame):
    print('Exiting EarModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__' or __name__.endswith('.ear_module'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    em = EarModule(lr=args.lr)
    em.run()
