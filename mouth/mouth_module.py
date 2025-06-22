"""
MouthModule: Synthesizes audio waveforms for each symbol using learnable parameters (no TTS).
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
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency
    sd = None
    print("[MouthModule] sounddevice not available, audio playback disabled")
from scipy.signal import sawtooth, square
from symbols import SYMBOLS
import atexit
import signal
import sys

# Config
HOST = '127.0.0.1'
PORT = 5004
SAMPLE_RATE = 16000
DURATION = 0.5  # seconds
MEMORY_PATH = 'mouth_params.npz'

try:
    from mouth.mouth_module import MouthModule
except ImportError:
    pass  # No internal imports needed, placeholder for consistency

class MouthModule:
    def __init__(self, lr=0.01):
        self.n_symbols = len(SYMBOLS)
        # Each symbol has its own learnable frequency, amplitude, and waveform type
        self.freqs = np.random.uniform(200, 800, self.n_symbols)
        self.amps = np.random.uniform(0.2, 0.8, self.n_symbols)
        self.wave_types = np.random.choice(['sine', 'square', 'sawtooth'], self.n_symbols)
        self.lr = lr
        self.symbols = SYMBOLS
        # Load memory if exists
        if os.path.exists(MEMORY_PATH):
            data = np.load(MEMORY_PATH, allow_pickle=True)
            self.freqs = data['freqs']
            self.amps = data['amps']
            self.wave_types = data['wave_types']
        atexit.register(self.save_memory)

    def save_memory(self):
        np.savez(MEMORY_PATH, freqs=self.freqs, amps=self.amps, wave_types=self.wave_types)

    def synthesize(self, symbol_idx):
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        freq = self.freqs[symbol_idx]
        amp = self.amps[symbol_idx]
        wave_type = self.wave_types[symbol_idx]
        if wave_type == 'sine':
            audio = amp * np.sin(2 * np.pi * freq * t)
        elif wave_type == 'square':
            audio = amp * square(2 * np.pi * freq * t)
        else:
            audio = amp * sawtooth(2 * np.pi * freq * t)
        return audio.astype(np.float32)

    def update(self, symbol_idx, feedback_audio):
        # Simple Delta Rule: adjust frequency and amplitude to minimize MSE
        pred_audio = self.synthesize(symbol_idx)
        error = feedback_audio - pred_audio
        grad_freq = np.mean(error * 2 * np.pi * self.amps[symbol_idx] * np.cos(2 * np.pi * self.freqs[symbol_idx] * np.linspace(0, DURATION, len(pred_audio), endpoint=False)) * np.linspace(0, DURATION, len(pred_audio), endpoint=False))
        grad_amp = np.mean(error * np.sin(2 * np.pi * self.freqs[symbol_idx] * np.linspace(0, DURATION, len(pred_audio), endpoint=False)))
        self.freqs[symbol_idx] += self.lr * grad_freq
        self.amps[symbol_idx] += self.lr * grad_amp
        return error

    def play(self, audio):
        if sd is None:
            print("[MouthModule] sounddevice unavailable, cannot play audio")
            return
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[MouthModule] Listening on {HOST}:{PORT}")
            while True:
                conn, addr = s.accept()
                with conn:
                    try:
                        data = b''
                        while True:
                            packet = conn.recv(4096)
                            if not packet:
                                break
                            data += packet
                        if not data:
                            continue
                        msg = pickle.loads(data)
                        if msg['cmd'] == 'speak':
                            symbol_idx = msg['symbol_idx']
                            if 0 <= symbol_idx < self.n_symbols:
                                audio = self.synthesize(symbol_idx)
                            else:
                                print(f"[MouthModule] Invalid symbol_idx: {symbol_idx}")
                                audio = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
                            response = {'audio': audio}
                        elif msg['cmd'] == 'learn':
                            symbol_idx = msg['symbol_idx']
                            feedback_audio = msg['feedback_audio']
                            if 0 <= symbol_idx < self.n_symbols:
                                error = self.update(symbol_idx, feedback_audio)
                            else:
                                print(f"[MouthModule] Invalid symbol_idx for learn: {symbol_idx}")
                                error = np.zeros(int(SAMPLE_RATE * DURATION))
                            response = {'error': error}
                        elif msg['cmd'] == 'play':
                            audio = msg['audio']
                            self.play(audio)
                            response = {'status': 'played'}
                        else:
                            response = {'error': 'Unknown command'}
                        conn.sendall(pickle.dumps(response))
                    except Exception as e:
                        print(f"[MouthModule] Error: {e}")
                        try:
                            conn.sendall(pickle.dumps({'error': str(e)}))
                        except Exception as send_err:
                            print(f"[MouthModule] Failed to send error response: {send_err}")

def signal_handler(sig, frame):
    print('Exiting MouthModule...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    mm = MouthModule(lr=args.lr)
    mm.run()
