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
import logging
import argparse
from typing import Optional
import asyncio
import json
import time
import uuid
from functools import partial

# Data augmentation utilities for images
from PIL import ImageEnhance, ImageOps

# Config
HOST = '127.0.0.1'
PORT = 5001
INPUT_SHAPE = (28, 28)
MEMORY_PATH = 'vision_weights.npy'

# Set up logging to both terminal and vision.log (overwrite on each run)
logger = logging.getLogger("VisionModule")
logger.setLevel(logging.INFO)
log_path = os.path.join(os.path.dirname(__file__), '../vision.log')
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

def augment_image(img, rotation=15, scale=0.1, noise=0.05, brightness=0.2):
    # Random rotation
    angle = np.random.uniform(-rotation, rotation)
    img = img.rotate(angle)
    # Random scaling
    scale_factor = 1 + np.random.uniform(-scale, scale)
    w, h = img.size
    img = img.resize((int(w * scale_factor), int(h * scale_factor)), Image.BILINEAR)
    img = img.resize((w, h), Image.BILINEAR)
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1 + np.random.uniform(-brightness, brightness))
    # Random noise
    arr = np.array(img).astype(np.float32) / 255.0
    arr += np.random.normal(0, noise, arr.shape)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

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

class DigitalRetina:
    """
    Simulates a digital retina (field of view) that can move over a larger image.
    Handles centering/padding if image is smaller than FOV.
    """
    def __init__(self, fov_shape=(28, 28), image_shape=(128, 128)):
        self.fov_shape = fov_shape
        self.image_shape = image_shape
        self.x = 0  # top-left x of FOV
        self.y = 0  # top-left y of FOV

    def set_position(self, x, y):
        self.x = int(np.clip(x, 0, max(0, self.image_shape[0] - self.fov_shape[0])))
        self.y = int(np.clip(y, 0, max(0, self.image_shape[1] - self.fov_shape[1])))

    def get_view(self, image):
        # image: 2D numpy array (image_shape)
        h, w = image.shape
        fh, fw = self.fov_shape
        # If image is smaller than FOV, pad and center
        if h < fh or w < fw:
            pad_h = max(0, fh - h)
            pad_w = max(0, fw - w)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
            h, w = image.shape
        x, y = self.x, self.y
        x = int(np.clip(x, 0, h - fh))
        y = int(np.clip(y, 0, w - fw))
        return image[x:x+fh, y:y+fw]

class ModuleClient:
    """
    Simple client for sending messages to other modules (mutual communication).
    Usage: client = ModuleClient(host, port); resp = client.send({'cmd': ..., ...})
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
    def send(self, msg):
        import socket, pickle, struct
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            data = pickle.dumps(msg)
            s.sendall(struct.pack('>I', len(data)) + data)
            header = s.recv(4)
            if not header:
                return None
            msg_len = struct.unpack('>I', header)[0]
            resp = b''
            while len(resp) < msg_len:
                packet = s.recv(msg_len - len(resp))
                if not packet:
                    break
                resp += packet
            return pickle.loads(resp)

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

# Example message schema generator

def make_message(src, dst, cmd, payload=None):
    return {
        'msg_id': str(uuid.uuid4()),
        'timestamp': time.time(),
        'src': src,
        'dst': dst,
        'cmd': cmd,
        'payload': payload or {},
    }

# Example: Add a handler registry to VisionModule
class VisionModule:
    def __init__(self, learning_rule='delta', lr=0.1, retina_fov=(28,28), retina_img_shape=(128,128)):
        # Log the initialization parameters
        logger.info(f"[VisionModule] __init__ called with learning_rule={learning_rule}, lr={lr}, retina_fov={retina_fov}, retina_img_shape={retina_img_shape}")
        # If FOV is smaller than image, use as given. If not, set FOV to image size so retina sees whole image.
        if retina_fov[0] < retina_img_shape[0] or retina_fov[1] < retina_img_shape[1]:
            self.retina_fov = retina_img_shape
            self.fov_covers_whole_image = True
        else:
            self.retina_fov = retina_fov
            self.fov_covers_whole_image = False
        self.input_dim = self.retina_fov[0] * self.retina_fov[1]
        self.output_dim = len(SYMBOLS)
        logger.info(f"[VisionModule] input_dim={self.input_dim}, output_dim={self.output_dim}")
        if learning_rule == 'delta':
            self.learner = DeltaRuleLearner(self.input_dim, self.output_dim, lr)
        else:
            self.learner = RescorlaWagnerLearner(self.input_dim, self.output_dim, lr)
        self.symbols = SYMBOLS
        self.retina = DigitalRetina(fov_shape=self.retina_fov, image_shape=retina_img_shape)
        # Load memory if exists, but check shape
        if os.path.exists(MEMORY_PATH):
            try:
                W = np.load(MEMORY_PATH)
                if W.shape == (self.output_dim, self.input_dim):
                    self.learner.W = W
                    logger.info(f"[VisionModule] Loaded weights from {MEMORY_PATH} with shape {W.shape}")
                else:
                    logger.warning(f"[VisionModule] Existing weights shape {W.shape} does not match expected {(self.output_dim, self.input_dim)}. Initializing new weights.")
            except Exception as e:
                logger.warning(f"[VisionModule] Could not load weights: {e}. Initializing new weights.")
        atexit.register(self.save_memory)

    def save_memory(self):
        np.save(MEMORY_PATH, self.learner.W)

    def process_image(self, img_path):
        img = Image.open(img_path).convert('L').resize(INPUT_SHAPE)
        arr = np.array(img).astype(np.float32) / 255.0
        return arr.flatten()

    def set_retina_position(self, x, y):
        self.retina.set_position(x, y)

    def notify_hand_module(self, symbol, hand_host='127.0.0.1', hand_port=5002, fov_pos=None, fov_image=None):
        """
        Notify the Hand module to draw a symbol after recognition.
        Always pass FOV position and optionally the FOV image for redraw/check.
        """
        client = ModuleClient(hand_host, hand_port)
        msg = {'cmd': 'draw', 'symbol': symbol}
        if fov_pos is not None:
            msg['fov_pos'] = fov_pos
        if fov_image is not None:
            # Send the FOV image as a list for serialization
            msg['fov_image'] = fov_image.tolist()
        try:
            resp = client.send(msg)
            logger.info(f"[VisionModule] Notified Hand module: {resp}")
            return resp
        except Exception as e:
            logger.error(f"[VisionModule] Failed to notify Hand module: {e}")
            return None

    def recognize(self, image, notify_hand=False, hand_host='127.0.0.1', hand_port=5002, pass_fov_pos=True):
        # image: 2D numpy array (retina_img_shape)
        logger.info(f"[VisionModule] recognize called with image shape: {getattr(image, 'shape', None)}")
        img_shape = image.shape
        x, y = self._random_retina_position(img_shape)
        self.retina.set_position(x, y)
        fov = self.retina.get_view(image)
        logger.info(f"[VisionModule] FOV shape: {fov.shape}, FOV pos: ({x},{y})")
        x_flat = fov.astype(np.float32).flatten()  # No normalization
        logger.info(f"[VisionModule] x_flat shape: {x_flat.shape}, W shape: {self.learner.W.shape}")
        y_pred = self.learner.predict(x_flat)
        idx = np.argmax(y_pred)
        symbol = self.symbols[idx]
        if notify_hand:
            self.notify_hand_module(symbol, hand_host, hand_port, fov_pos=(x, y), fov_image=fov)
        return symbol, y_pred, (x, y)

    def learn(self, image, target_idx, log_fov=True, epoch=None, batch_idx=None, sample_idx=None, save_debug=False):
        logger.info(f"[VisionModule] learn called with image shape: {getattr(image, 'shape', None)}, target_idx: {target_idx}")
        img_shape = image.shape
        x, y = self._random_retina_position(img_shape)
        self.retina.set_position(x, y)
        fov = self.retina.get_view(image)
        logger.info(f"[VisionModule] FOV shape: {fov.shape}, FOV pos: ({x},{y})")
        if log_fov:
            logger.info(f"[VisionModule] FOV pos for sample: x={x}, y={y}, img_shape={img_shape}, fov_shape={self.retina.fov_shape}")
        if Image is not None:
            img_aug = augment_image(Image.fromarray((fov * 255).astype(np.uint8)))
            fov = np.array(img_aug).astype(np.float32) / 255.0
        x_flat = fov.flatten()  # No normalization
        logger.info(f"[VisionModule] x_flat shape: {x_flat.shape}, W shape: {self.learner.W.shape}")
        t = np.zeros(self.output_dim)
        t[target_idx] = 1.0
        pred = self.learner.predict(x_flat)
        logger.info(f"[VisionModule] Prediction: {pred}, Target: {t}")
        error = self.learner.update(x_flat, t)
        logger.info(f"[VisionModule] Weight update error: {error}")
        if save_debug and epoch is not None and batch_idx is not None and sample_idx is not None:
            self._save_debug_fov(fov, epoch, batch_idx, sample_idx)
        return error

    def batch_learn(self, X, target_indices, batch_size=8, augment=True, epoch=None):
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        batch_losses = []
        for batch_num, start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            batch_X = []
            batch_targets = []
            batch_fovs = []
            batch_positions = []
            batch_target_values = []
            batch_img_summaries = []
            for i, idx in enumerate(batch_idx):
                x = X[idx]
                target = target_indices[idx]
                batch_target_values.append(target)
                # Log a summary of the image (min, max, mean, std)
                img_summary = {
                    'min': float(np.min(x)),
                    'max': float(np.max(x)),
                    'mean': float(np.mean(x)),
                    'std': float(np.std(x))
                }
                batch_img_summaries.append(img_summary)
                pos_x, pos_y = self._random_retina_position(x.shape)
                self.retina.set_position(pos_x, pos_y)
                fov = self.retina.get_view(x)
                batch_positions.append((pos_x, pos_y))
                if Image is not None:
                    img_aug = augment_image(Image.fromarray((fov * 255).astype(np.uint8)))
                    fov = np.array(img_aug).astype(np.float32) / 255.0
                batch_fovs.append(fov)
                x_flat = fov.flatten()  # No normalization
                batch_X.append(x_flat)
                t = np.zeros(self.output_dim)
                t[target] = 1.0
                batch_targets.append(t)
                if epoch is not None and np.random.rand() < 0.05:
                    self._save_debug_fov(fov, epoch, batch_num, i)
            logger.info(f"[VisionModule] Batch FOV positions: {batch_positions}")
            logger.info(f"[VisionModule] Batch targets: {batch_target_values}, unique: {set(batch_target_values)}")
            logger.info(f"[VisionModule] Batch image summaries: {batch_img_summaries}")
            batch_X = np.stack(batch_X)
            batch_targets = np.stack(batch_targets)
            batch_error = []
            for x, t in zip(batch_X, batch_targets):
                pred = self.learner.predict(x)
                logger.info(f"[VisionModule] Batch Prediction: {pred}, Target: {t}")
                error = self.learner.update(x, t)
                logger.info(f"[VisionModule] Batch Weight update error: {error}")
                batch_error.append(np.mean(np.abs(error)))
            batch_losses.append(np.mean(batch_error))
        return batch_losses

    def train_with_scheduler(self, X, target_indices, batch_size=8, augment=True, max_epochs=50, early_stop_patience=7, min_delta=1e-4, lr_scheduler=None, val_split=0.1, log_metrics_path=None):
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
            train_losses = self.batch_learn(X_train, y_train, batch_size, augment, epoch=epoch)
            train_loss = np.mean(train_losses)
            val_losses = self.batch_learn(X_val, y_val, batch_size, False, epoch=epoch) if val_size > 0 else [0]
            val_loss = np.mean(val_losses)
            # LR scheduling
            if lr_scheduler:
                lr = lr_scheduler.step(val_loss)
                self.learner.lr = lr
            else:
                lr = self.learner.lr
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr)
            logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}")
            # Log weight matrix stats
            W = self.learner.W
            logger.info(f"[VisionModule] Weights stats after epoch {epoch+1}: min={W.min():.4f}, max={W.max():.4f}, mean={W.mean():.4f}, std={W.std():.4f}")
            # Save metrics
            if log_metrics_path:
                with open(log_metrics_path, 'a') as f:
                    f.write(f"{epoch+1},{train_loss},{val_loss},{lr}\n")
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
            elif epoch - best_epoch >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        return history

    def run(self):
        import threading, time
        def heartbeat():
            while True:
                logger.info("[Heartbeat] VisionModule is alive and listening...")
                time.sleep(30)
        threading.Thread(target=heartbeat, daemon=True).start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[VisionModule] Listening on {HOST}:{PORT}")  # <--- Added print for immediate feedback
            logger.info(f"Listening on {HOST}:{PORT}")
            while True:
                try:
                    conn, addr = s.accept()
                    logger.info(f"Accepted connection from {addr}")
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
                        try:
                            if msg['cmd'] == 'recognize':
                                x = msg['data']
                                logger.info(f"Processing recognize command. Data type: {type(x)}, shape: {getattr(x, 'shape', None)}")
                                label, y = self.recognize(x)
                                response = {'label': label, 'y': y}
                            elif msg['cmd'] == 'learn':
                                x = msg['data']
                                target_idx = msg['target_idx']
                                # Support both single and batch learning
                                if isinstance(x, np.ndarray) and x.ndim == 2:
                                    # Batch mode
                                    logger.info(f"[VisionModule] Batch learn detected in 'learn' command. Shape: {x.shape}")
                                    errors = self.batch_learn(x, target_idx, batch_size=8, augment=True)
                                    response = {'status': 'ok', 'errors': errors}
                                else:
                                    # Single sample mode
                                    logger.info(f"[VisionModule] Single sample learn detected in 'learn' command.")
                                    error = self.learn(x, target_idx)
                                    response = {'status': 'ok', 'error': error}
                            elif msg['cmd'] == 'shutdown':
                                logger.info('Shutdown command received. Exiting...')
                                response = {'status': 'shutting down'}
                                resp = pickle.dumps(response)
                                resp_len = struct.pack('>I', len(resp))
                                conn.sendall(resp_len + resp)
                                sys.exit(0)
                            else:
                                response = {'error': 'Unknown command'}
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            response = {'error': str(e)}
                        # Send response
                        resp_data = pickle.dumps(response)
                        resp_len = struct.pack('>I', len(resp_data))
                        conn.sendall(resp_len + resp_data)
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")

    def _random_retina_position(self, img_shape):
        """
        Returns a valid (x, y) position for the retina FOV within the image.
        If FOV covers the whole image, always return (0, 0).
        Otherwise, pick a random valid position.
        """
        if self.fov_covers_whole_image:
            return 0, 0
        h, w = img_shape[:2]
        fh, fw = self.retina_fov
        max_x = max(0, h - fh)
        max_y = max(0, w - fw)
        x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        return x, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start VisionModule server.")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--rule', type=str, default='delta', choices=['delta', 'rescorla'], help='Learning rule')
    parser.add_argument('--fov', type=int, nargs=2, default=[28, 28], help='Retina field of view (h w)')
    parser.add_argument('--img-shape', type=int, nargs=2, default=[128, 128], help='Input image shape (h w)')
    args = parser.parse_args()

    vm = VisionModule(learning_rule=args.rule, lr=args.lr, retina_fov=tuple(args.fov), retina_img_shape=tuple(args.img_shape))
    print(f"[VisionModule] Starting with lr={args.lr}, rule={args.rule}, fov={args.fov}, img_shape={args.img_shape}")
    logger.info(f"[VisionModule] Starting with lr={args.lr}, rule={args.rule}, fov={args.fov}, img_shape={args.img_shape}")
    vm.run()
