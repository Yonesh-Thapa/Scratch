"""
VisionModule: Recognizes symbols from images using Delta Rule/Rescorla-Wagner learning.
Runs as an independent process, communicates via sockets/IPC.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import os
try:
    import numpy as np
except ImportError:
    print("[VisionModule] numpy is required but not installed")
    sys.exit(1)
import socket
import pickle
try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    print("[VisionModule] PIL not available, image loading disabled")
from symbols import SYMBOLS
try:
    from vision.delta_rule import DeltaRuleLearner
    from vision.rescorla_wagner import RescorlaWagnerLearner
except ImportError:
    from delta_rule import DeltaRuleLearner
    from rescorla_wagner import RescorlaWagnerLearner
import atexit
import signal
import sys
import struct

# Config
HOST = '127.0.0.1'
PORT = 5001
INPUT_SHAPE = (28, 28)
MEMORY_PATH = 'vision_weights.npy'

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

class VisionModule:
    def __init__(self, learning_rule='delta', lr=0.01):
        self.input_dim = INPUT_SHAPE[0] * INPUT_SHAPE[1]
        self.output_dim = len(SYMBOLS)
        if learning_rule == 'delta':
            self.learner = DeltaRuleLearner(self.input_dim, self.output_dim, lr)
        else:
            self.learner = RescorlaWagnerLearner(self.input_dim, self.output_dim, lr)
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.learner.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def save_memory(self):
        np.save(MEMORY_PATH, self.learner.W)

    def process_image(self, img_path):
        img = Image.open(img_path).convert('L').resize(INPUT_SHAPE)
        arr = np.array(img).astype(np.float32) / 255.0
        return arr.flatten()

    def recognize(self, x):
        y = self.learner.predict(x)
        idx = np.argmax(y)
        return self.symbols[idx], y

    def learn(self, x, target_idx):
        target = np.zeros(self.output_dim)
        target[target_idx] = 1.0
        error = self.learner.update(x, target)
        return error

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[VisionModule] Listening on {HOST}:{PORT}")
            while True:
                try:
                    conn, addr = s.accept()
                    print(f"[VisionModule] Accepted connection from {addr}")
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
                            print(f"[VisionModule] Decoded message: {msg}")
                        except Exception as e:
                            print(f"[VisionModule] Error decoding message: {e}")
                            resp = pickle.dumps({'error': f'Error decoding message: {e}'})
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            continue
                        try:
                            if msg['cmd'] == 'recognize':
                                x = msg['data']
                                print(f"[VisionModule] Processing recognize command. Data type: {type(x)}, shape: {getattr(x, 'shape', None)}")
                                label, y = self.recognize(x)
                                response = {'label': label, 'y': y}
                            elif msg['cmd'] == 'learn':
                                x = msg['data']
                                target_idx = msg['target_idx']
                                print(f"[VisionModule] Processing learn command. Data type: {type(x)}, shape: {getattr(x, 'shape', None)}, target_idx: {target_idx}")
                                error = self.learn(x, target_idx)
                                response = {'error': error}
                            elif msg['cmd'] == 'shutdown':
                                print('[VisionModule] Shutdown command received. Exiting...')
                                response = {'status': 'shutting down'}
                                resp = pickle.dumps(response)
                                resp_len = struct.pack('>I', len(resp))
                                conn.sendall(resp_len + resp)
                                sys.exit(0)
                            else:
                                print(f"[VisionModule] Unknown command: {msg['cmd']}")
                                response = {'error': f'Unknown command: {msg["cmd"]}'}
                        except Exception as e:
                            print(f"[VisionModule] Error processing command: {e}")
                            response = {'error': f'Error processing command: {e}'}
                        resp = pickle.dumps(response)
                        resp_len = struct.pack('>I', len(resp))
                        conn.sendall(resp_len + resp)
                except Exception as e:
                    print(f"[VisionModule] Connection error: {e}")

def signal_handler(sig, frame):
    print('Exiting VisionModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__' or __name__.endswith('.vision_module'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule', choices=['delta', 'rescorla'], default='delta')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    vm = VisionModule(learning_rule=args.rule, lr=args.lr)
    vm.run()
