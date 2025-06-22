"""
MouthModule: Synthesizes audio (waveforms) for symbols, receives feedback to update.
Runs as an independent process, communicates via sockets/IPC.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import os
try:
    import numpy as np
except ImportError:
    print("[MouthModule] numpy is required but not installed")
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
PORT = 5004
MEMORY_PATH = 'mouth_weights.npy'
WAVE_DIM = 8000  # E.g., 0.5 sec at 16kHz, or use your default

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

class MouthModule:
    def __init__(self, lr=0.01):
        self.output_dim = len(SYMBOLS)
        self.wave_dim = WAVE_DIM
        self.W = np.random.randn(self.output_dim, self.wave_dim) * 0.01
        self.lr = lr
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def synthesize(self, symbol_idx):
        return self.W[symbol_idx]

    def update(self, symbol_idx, feedback_wave):
        # Normalize feedback
        feedback_wave = (feedback_wave - np.mean(feedback_wave)) / (np.std(feedback_wave) + 1e-8)
        error = feedback_wave - self.W[symbol_idx]
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
            print(f"[MouthModule] Listening on {HOST}:{PORT}")
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
                            print(f"[MouthModule] Decoded message: {msg}")
                        except Exception as e:
                            print(f"[MouthModule] Error decoding message: {e}")
                            resp = pickle.dumps({'error': f'Error decoding message: {e}'})
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            continue
                        if msg['cmd'] == 'speak':
                            symbol_idx = msg['symbol_idx']
                            print(f"[MouthModule] Processing speak command. symbol_idx: {symbol_idx}")
                            if 0 <= symbol_idx < len(self.symbols):
                                wave = self.synthesize(symbol_idx)
                            else:
                                print(f"[MouthModule] Invalid symbol_idx: {symbol_idx}")
                                wave = np.zeros(self.wave_dim, dtype=np.float32)
                            response = {'wave': wave}
                        elif msg['cmd'] == 'learn':
                            symbol_idx = msg['symbol_idx']
                            feedback_wave = msg['feedback_wave']
                            print(f"[MouthModule] Processing learn command. symbol_idx: {symbol_idx}, feedback_wave type: {type(feedback_wave)}, shape: {getattr(feedback_wave, 'shape', None)}")
                            if 0 <= symbol_idx < len(self.symbols):
                                error = self.update(symbol_idx, feedback_wave)
                            else:
                                print(f"[MouthModule] Invalid symbol_idx for learn: {symbol_idx}")
                                error = np.zeros(self.wave_dim)
                            response = {'error': error}
                        elif msg['cmd'] == 'shutdown':
                            print('[MouthModule] Shutdown command received. Exiting...')
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
                    print(f"[MouthModule] Connection error: {e}")

def signal_handler(sig, frame):
    print('Exiting MouthModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__' or __name__.endswith('.mouth_module'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    mm = MouthModule(lr=args.lr)
    mm.run()
