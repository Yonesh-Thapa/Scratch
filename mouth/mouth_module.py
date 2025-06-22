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
import logging
import argparse
from typing import Optional
import asyncio
import json
import time
import uuid
from functools import partial

# Config
HOST = '127.0.0.1'
PORT = 5004
MEMORY_PATH = 'mouth_weights.npy'
WAVE_DIM = 8000  # E.g., 0.5 sec at 16kHz, or use your default

# Set up logging to both terminal and mouth.log (overwrite on each run)
logger = logging.getLogger("MouthModule")
logger.setLevel(logging.INFO)
log_path = os.path.join(os.path.dirname(__file__), '../mouth.log')
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers = [file_handler, stream_handler]

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# Xavier/He initialization utility
def xavier_init(shape):
    fan_in, fan_out = shape[1], shape[0]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

# Adaptive learning rate scheduler
class LRScheduler:
    def __init__(self, lr, factor=0.5, patience=5, min_lr=1e-5):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self.wait = 0
        return self.lr

class DigitalArticulator:
    """
    Simulates a digital mouth that generates waveforms from control parameters.
    """
    def __init__(self, wave_dim=8000, sr=16000, noise_std=0.01):
        self.wave_dim = wave_dim
        self.sr = sr
        self.noise_std = noise_std

    def synthesize(self, controls):
        # controls: dict with keys like 'pitch', 'amplitude', 'formant1', 'formant2', ...
        t = np.linspace(0, self.wave_dim / self.sr, self.wave_dim)
        # Simple synthesis: sum of sinusoids (pitch + formants)
        pitch = controls.get('pitch', 220)
        amp = controls.get('amplitude', 0.5)
        f1 = controls.get('formant1', 700)
        f2 = controls.get('formant2', 1200)
        wave = amp * (np.sin(2 * np.pi * pitch * t) + 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t))
        # Add noise
        wave += np.random.normal(0, self.noise_std, wave.shape)
        return wave.astype(np.float32)

class AsyncModuleServer:
    """
    Asynchronous server for handling incoming JSON messages from any peer module.
    Extensible with custom command handlers.
    """
    def __init__(self, host, port, module_name, handler_registry=None, logger=None):
        self.host = host
        self.port = port
        self.module_name = module_name
        self.handler_registry = handler_registry or {}
        self.logger = logger or logging.getLogger(module_name)
        self.server = None

    async def handle_client(self, reader, writer):
        try:
            data = await reader.read(4)
            if not data or len(data) < 4:
                writer.close()
                await writer.wait_closed()
                return
            msg_len = int.from_bytes(data, 'big')
            msg_data = await reader.read(msg_len)
            msg = json.loads(msg_data.decode('utf-8'))
            self.logger.info(f"[AsyncServer] Received: {msg}")
            cmd = msg.get('cmd')
            handler = self.handler_registry.get(cmd)
            if handler:
                resp = await handler(msg)
            else:
                resp = {'error': f'Unknown command: {cmd}'}
            resp_bytes = json.dumps(resp).encode('utf-8')
            writer.write(len(resp_bytes).to_bytes(4, 'big') + resp_bytes)
            await writer.drain()
        except Exception as e:
            self.logger.error(f"[AsyncServer] Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        self.logger.info(f"[AsyncServer] Listening on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

class AsyncModuleClient:
    """
    Asynchronous client for sending JSON messages to any peer module.
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("AsyncModuleClient")

    async def send(self, host, port, msg, timeout=5):
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            msg_bytes = json.dumps(msg).encode('utf-8')
            writer.write(len(msg_bytes).to_bytes(4, 'big') + msg_bytes)
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4), timeout=timeout)
            if not data or len(data) < 4:
                writer.close()
                await writer.wait_closed()
                return None
            msg_len = int.from_bytes(data, 'big')
            resp_data = await asyncio.wait_for(reader.read(msg_len), timeout=timeout)
            resp = json.loads(resp_data.decode('utf-8'))
            writer.close()
            await writer.wait_closed()
            self.logger.info(f"[AsyncClient] Sent to {host}:{port}, got: {resp}")
            return resp
        except Exception as e:
            self.logger.error(f"[AsyncClient] Error: {e}")
            return {'error': str(e)}

def make_message(src, dst, cmd, payload=None):
    return {
        'msg_id': str(uuid.uuid4()),
        'timestamp': time.time(),
        'src': src,
        'dst': dst,
        'cmd': cmd,
        'payload': payload or {},
    }

class MouthModule:
    def __init__(self, lr=0.01, wave_dim=WAVE_DIM, sr=16000):
        self.output_dim = len(SYMBOLS)
        self.wave_dim = wave_dim
        self.W = np.random.randn(self.output_dim, 4) * 0.01  # 4 control params: pitch, amp, f1, f2
        self.lr = lr
        self.symbols = SYMBOLS
        self.articulator = DigitalArticulator(wave_dim=wave_dim, sr=sr)
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def synthesize(self, symbol_idx):
        controls = self.W[symbol_idx]
        control_dict = {'pitch': controls[0]*200+220, 'amplitude': np.clip(controls[1], 0.1, 1.0), 'formant1': controls[2]*500+700, 'formant2': controls[3]*500+1200}
        return self.articulator.synthesize(control_dict)

    def update(self, symbol_idx, feedback_wave):
        # Estimate controls from feedback_wave (placeholder: not invertible, so use dummy gradient)
        # In practice, use a differentiable loss between synthesized and feedback wave
        pred_wave = self.synthesize(symbol_idx)
        error = feedback_wave[:self.wave_dim] - pred_wave
        # Simple gradient: update controls in direction that reduces error norm
        grad = np.random.randn(4) * np.mean(error) * 0.001  # Placeholder gradient
        self.W[symbol_idx] += self.lr * grad
        self.W[symbol_idx] = np.clip(self.W[symbol_idx], -10, 10)
        logger.info(f"[MouthModule] Updated controls for symbol {symbol_idx}: {self.W[symbol_idx]}")
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)

    def run(self):
        import threading, time
        def heartbeat():
            while True:
                logger.info("[Heartbeat] MouthModule is alive and listening...")
                time.sleep(30)
        threading.Thread(target=heartbeat, daemon=True).start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            logger.info(f"[MouthModule] Listening on {HOST}:{PORT}")
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
                            logger.info(f"[MouthModule] Decoded message: {msg}")
                        except Exception as e:
                            logger.error(f"[MouthModule] Error decoding message: {e}")
                            resp = pickle.dumps({'error': f'Error decoding message: {e}'})
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            continue
                        if msg['cmd'] == 'speak':
                            symbol_idx = msg['symbol_idx']
                            logger.info(f"[MouthModule] Processing speak command. symbol_idx: {symbol_idx}")
                            if 0 <= symbol_idx < len(self.symbols):
                                wave = self.synthesize(symbol_idx)
                            else:
                                logger.error(f"[MouthModule] Invalid symbol_idx: {symbol_idx}")
                                wave = np.zeros(self.wave_dim, dtype=np.float32)
                            response = {'wave': wave}
                        elif msg['cmd'] == 'learn':
                            symbol_idx = msg['symbol_idx']
                            feedback_wave = msg['feedback_wave']
                            logger.info(f"[MouthModule] Processing learn command. symbol_idx: {symbol_idx}, feedback_wave type: {type(feedback_wave)}, shape: {getattr(feedback_wave, 'shape', None)}")
                            if 0 <= symbol_idx < len(self.symbols):
                                error = self.update(symbol_idx, feedback_wave)
                            else:
                                logger.error(f"[MouthModule] Invalid symbol_idx for learn: {symbol_idx}")
                                error = np.zeros(self.wave_dim)
                            response = {'error': error}
                        elif msg['cmd'] == 'shutdown':
                            logger.info('[MouthModule] Shutdown command received. Exiting...')
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
                    logger.error(f"[MouthModule] Connection error: {e}")

    def register_async_handlers(self):
        self.async_handlers = {
            'speak': self.async_handle_speak,
            'learn': self.async_handle_learn,
            # Add more handlers as needed
        }

    async def async_handle_speak(self, msg):
        symbol_idx = msg['payload']['symbol_idx']
        wave = self.synthesize(symbol_idx)
        return {'wave': wave.tolist() if hasattr(wave, 'tolist') else wave}

    async def async_handle_learn(self, msg):
        symbol_idx = msg['payload']['symbol_idx']
        feedback_wave = np.array(msg['payload']['feedback_wave'])
        error = self.update(symbol_idx, feedback_wave)
        return {'error': error.tolist() if hasattr(error, 'tolist') else error}

    def start_async_server(self, host=HOST, port=PORT):
        self.register_async_handlers()
        server = AsyncModuleServer(host, port, 'MouthModule', self.async_handlers, logger)
        asyncio.create_task(server.start())
        logger.info("[MouthModule] Async server started.")

    async def async_notify_peer(self, peer_host, peer_port, cmd, payload=None, dst='peer'):
        client = AsyncModuleClient(logger)
        msg = make_message('mouth', dst, cmd, payload)
        return await client.send(peer_host, peer_port, msg)

    def batch_learn(self, X, target_indices, batch_size=8, augment=False):
        """
        Batch/mini-batch training for MouthModule.
        X: list/array of feedback waveforms
        target_indices: list of target indices
        batch_size: number of samples per batch
        augment: whether to apply augmentation (not implemented for mouth)
        Returns: list of batch losses
        """
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        batch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            batch_X = [X[i] for i in batch_idx]
            batch_targets = [target_indices[i] for i in batch_idx]
            batch_error = []
            for wave, idx in zip(batch_X, batch_targets):
                error = self.update(idx, wave)
                batch_error.append(np.mean(np.abs(error)))
            batch_losses.append(np.mean(batch_error))
        return batch_losses

    def train_with_scheduler(self, X, target_indices, batch_size=8, max_epochs=50, early_stop_patience=7, min_delta=1e-4, lr_scheduler=None, val_split=0.1, log_metrics_path=None):
        n = len(X)
        val_size = int(n * val_split)
        train_idx = np.arange(n)
        np.random.shuffle(train_idx)
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]
        X_train = [X[i] for i in train_idx]
        y_train = [target_indices[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [target_indices[i] for i in val_idx]
        best_val_loss = float('inf')
        best_epoch = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        for epoch in range(max_epochs):
            train_losses = self.batch_learn(X_train, y_train, batch_size)
            train_loss = np.mean(train_losses)
            val_losses = self.batch_learn(X_val, y_val, batch_size) if val_size > 0 else [0]
            val_loss = np.mean(val_losses)
            if lr_scheduler:
                lr = lr_scheduler.step(val_loss)
                self.lr = lr
            else:
                lr = self.lr
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr)
            logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}")
            if log_metrics_path:
                with open(log_metrics_path, 'a') as f:
                    f.write(f"{epoch+1},{train_loss},{val_loss},{lr}\n")
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
            elif epoch - best_epoch >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        return history

def signal_handler(sig, frame):
    logger.info('Exiting MouthModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__' or __name__.endswith('.mouth_module'):
    parser = argparse.ArgumentParser(description="MouthModule: Production-grade audio synthesis with adaptive learning and more.")
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--early-stop', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Minimum improvement for early stopping')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split fraction')
    parser.add_argument('--scheduler', action='store_true', help='Enable adaptive learning rate scheduler')
    parser.add_argument('--log-metrics', type=str, default=None, help='Path to save training metrics (CSV)')
    args = parser.parse_args()

    mm = MouthModule(lr=args.lr)
    # For socket/IPC use, call mm.run()
    # For standalone training (example):
    # X, y = ... # Load your dataset here as list/array of waveforms and target indices
    # scheduler = LRScheduler(args.lr) if args.scheduler else None
    # mm.train_with_scheduler(X, y, batch_size=args.batch_size, max_epochs=args.epochs, early_stop_patience=args.early_stop, min_delta=args.min_delta, lr_scheduler=scheduler, val_split=args.val_split, log_metrics_path=args.log_metrics)
    mm.run()

# USAGE INSTRUCTIONS:
# - Run with python mouth_module.py --help for all options.
# - For production training, use the train_with_scheduler method (see code comments above).
# - For socket-based operation, use the run() method (default).
# - All logs and metrics are saved to mouth.log and (optionally) CSV.
# - See README for more details and integration examples.
