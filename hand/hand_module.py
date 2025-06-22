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
PORT = 5002
CANVAS_SIZE = (28, 28)
MEMORY_PATH = 'hand_weights.npy'

# Set up logging to both terminal and hand.log (overwrite on each run)
logger = logging.getLogger("HandModule")
logger.setLevel(logging.INFO)
log_path = os.path.join(os.path.dirname(__file__), '../hand.log')
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
        # Simple Bresenham's line algorithm for digital drawing
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

class HandModule:
    def __init__(self, lr=0.5):  # Increased default learning rate
        self.input_dim = np.prod(CANVAS_SIZE)
        self.output_dim = len(SYMBOLS)
        # Initialize weights with higher variance for more visible changes
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.2
        self.lr = lr
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            self.W = np.load(MEMORY_PATH)
        atexit.register(self.save_memory)

    def draw_symbol(self, symbol_idx, actions=None):
        """
        If actions is None, use stored weights to generate a default drawing.
        If actions is a list of pen actions, execute them on a digital canvas.
        """
        canvas = DigitalCanvas(size=CANVAS_SIZE)
        pen = DigitalPen(canvas, start_pos=(CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2))
        if actions is not None:
            for act in actions:
                if act['type'] == 'pen_down':
                    pen.pen_down()
                elif act['type'] == 'pen_up':
                    pen.pen_up()
                elif act['type'] == 'move':
                    pen.move_to(act['x'], act['y'], value=1.0, thickness=1)
        else:
            # Default: interpret weights as a raster image
            return self.W[symbol_idx].reshape(CANVAS_SIZE)
        return canvas.get_image()

    def update(self, symbol_idx, feedback_img):
        flat_feedback = feedback_img.flatten()
        # Normalize feedback
        flat_feedback = (flat_feedback - np.mean(flat_feedback)) / (np.std(flat_feedback) + 1e-8)
        pred = self.W[symbol_idx]
        error = flat_feedback - pred
        logger.info(f"[HandModule] symbol_idx={symbol_idx} pred_mean={np.mean(pred):.4f} feedback_mean={np.mean(flat_feedback):.4f} error_mean={np.mean(error):.4f} error_std={np.std(error):.4f}")
        self.W[symbol_idx] += self.lr * error
        # Clip weights
        self.W[symbol_idx] = np.clip(self.W[symbol_idx], -10, 10)
        logger.info(f"[HandModule] Updated W[{symbol_idx}] mean={np.mean(self.W[symbol_idx]):.4f} std={np.std(self.W[symbol_idx]):.4f}")
        return error

    def save_memory(self):
        np.save(MEMORY_PATH, self.W)

    def run(self):
        import threading, time
        def heartbeat():
            while True:
                logger.info("[Heartbeat] HandModule is alive and listening...")
                time.sleep(30)
        threading.Thread(target=heartbeat, daemon=True).start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            logger.info(f"Listening on {HOST}:{PORT}")
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
                            logger.info(f"Decoded message: {msg}")
                        except Exception as e:
                            logger.error(f"Error decoding message: {e}")
                            resp = pickle.dumps({'error': f'Error decoding message: {e}'})
                            resp_len = struct.pack('>I', len(resp))
                            conn.sendall(resp_len + resp)
                            continue
                        if msg['cmd'] == 'draw':
                            symbol_idx = msg['symbol_idx']
                            logger.info(f"Processing draw command. symbol_idx: {symbol_idx}")
                            if 0 <= symbol_idx < len(self.symbols):
                                img = self.draw_symbol(symbol_idx)
                            else:
                                logger.warning(f"Invalid symbol_idx: {symbol_idx}")
                                img = np.zeros(CANVAS_SIZE, dtype=np.float32)
                            response = {'img': img}
                        elif msg['cmd'] == 'learn':
                            symbol_idx = msg['symbol_idx']
                            feedback_img = msg['feedback_img']
                            logger.info(f"Processing learn command. symbol_idx: {symbol_idx}, feedback_img type: {type(feedback_img)}, shape: {getattr(feedback_img, 'shape', None)}")
                            if 0 <= symbol_idx < len(self.symbols):
                                error = self.update(symbol_idx, feedback_img)
                            else:
                                logger.warning(f"Invalid symbol_idx for learn: {symbol_idx}")
                                error = np.zeros(np.prod(CANVAS_SIZE))
                            response = {'error': error}
                        elif msg['cmd'] == 'redraw':
                            symbol_idx = msg['symbol_idx']
                            logger.info(f"Processing redraw command. symbol_idx: {symbol_idx}")
                            if 0 <= symbol_idx < len(self.symbols):
                                img = self.draw_symbol(symbol_idx)
                            else:
                                logger.warning(f"Invalid symbol_idx: {symbol_idx}")
                                img = np.zeros(CANVAS_SIZE, dtype=np.float32)
                            response = {'img': img}
                        elif msg['cmd'] == 'shutdown':
                            logger.info('Shutdown command received. Exiting...')
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
                    logger.error(f"Connection error: {e}")

    def batch_learn(self, X, target_indices, batch_size=8, augment=False):
        """
        Batch/mini-batch training for HandModule.
        X: list/array of input images (flattened)
        target_indices: list of target indices
        batch_size: number of samples per batch
        augment: whether to apply augmentation (not implemented for hand)
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
            for x, idx in zip(batch_X, batch_targets):
                error = self.update(idx, x.reshape(CANVAS_SIZE))
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

    def register_async_handlers(self):
        self.async_handlers = {
            'draw': self.async_handle_draw,
            'learn': self.async_handle_learn,
            # Add more handlers as needed
        }

    async def async_handle_draw(self, msg):
        symbol_idx = msg['payload']['symbol_idx']
        img = self.draw_symbol(symbol_idx)
        return {'img': img.tolist() if hasattr(img, 'tolist') else img}

    async def async_handle_learn(self, msg):
        symbol_idx = msg['payload']['symbol_idx']
        feedback_img = np.array(msg['payload']['feedback_img'])
        error = self.update(symbol_idx, feedback_img)
        return {'error': error.tolist() if hasattr(error, 'tolist') else error}

    def start_async_server(self, host=HOST, port=PORT):
        self.register_async_handlers()
        server = AsyncModuleServer(host, port, 'HandModule', self.async_handlers, logger)
        asyncio.create_task(server.start())
        logger.info("[HandModule] Async server started.")

    async def async_notify_peer(self, peer_host, peer_port, cmd, payload=None, dst='peer'):
        client = AsyncModuleClient(logger)
        msg = make_message('hand', dst, cmd, payload)
        return await client.send(peer_host, peer_port, msg)

    def check_similarity(self, img1, img2):
        # Utility: returns MSE similarity (negative value, higher is better)
        img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
        img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
        return -np.mean((img1.flatten() - img2.flatten())**2)

def signal_handler(sig, frame):
    logger.info('Exiting HandModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__' or __name__.endswith('.hand_module'):
    parser = argparse.ArgumentParser(description="HandModule: Production-grade symbol drawing with adaptive learning and more.")
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--early-stop', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Minimum improvement for early stopping')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split fraction')
    parser.add_argument('--scheduler', action='store_true', help='Enable adaptive learning rate scheduler')
    parser.add_argument('--log-metrics', type=str, default=None, help='Path to save training metrics (CSV)')
    args = parser.parse_args()

    hm = HandModule(lr=args.lr)
    # For socket/IPC use, call hm.run()
    # For standalone training (example):
    # X, y = ... # Load your dataset here as list/array of images and target indices
    # scheduler = LRScheduler(args.lr) if args.scheduler else None
    # hm.train_with_scheduler(X, y, batch_size=args.batch_size, max_epochs=args.epochs, early_stop_patience=args.early_stop, min_delta=args.min_delta, lr_scheduler=scheduler, val_split=args.val_split, log_metrics_path=args.log_metrics)
    hm.run()

# USAGE INSTRUCTIONS:
# - Run with python hand_module.py --help for all options.
# - For production training, use the train_with_scheduler method (see code comments above).
# - For socket-based operation, use the run() method (default).
# - All logs and metrics are saved to hand.log and (optionally) CSV.
# - See README for more details and integration examples.
