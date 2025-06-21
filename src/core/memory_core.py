"""
memory_core.py
Handles saving and loading of learned decoder, attention, and recognizer weights to separate files.
"""
import numpy as np
import os

class MemoryCore:
    """Saves and loads learned decoder, attention, and recognizer weights to separate files."""
    def __init__(self, decoder_path='decoder_weights.npy', attn_path='attention_weights.npy', recog_path='recognizer_prototypes.npy'):
        self.decoder_path = decoder_path
        self.attn_path = attn_path
        self.recog_path = recog_path

    def save_decoder(self, weights):
        np.save(self.decoder_path, weights)
        print(f"[MemoryCore] Saved decoder weights to {self.decoder_path}")

    def load_decoder(self):
        if os.path.exists(self.decoder_path):
            weights = np.load(self.decoder_path, allow_pickle=True)
            print(f"[MemoryCore] Loaded decoder weights from {self.decoder_path}")
            return weights
        else:
            print(f"[MemoryCore] No saved weights found at {self.decoder_path}")
            return None

    def save_attention(self, W, b):
        np.save(self.attn_path, {'W': W, 'b': b})
        print(f"[MemoryCore] Saved attention weights to {self.attn_path}")

    def load_attention(self):
        if os.path.exists(self.attn_path):
            data = np.load(self.attn_path, allow_pickle=True).item()
            print(f"[MemoryCore] Loaded attention weights from {self.attn_path}")
            return data['W'], data['b']
        else:
            print(f"[MemoryCore] No saved attention weights found at {self.attn_path}")
            return None, None

    def save_recognizer(self, prototypes):
        np.save(self.recog_path, prototypes)
        print(f"[MemoryCore] Saved recognizer prototypes to {self.recog_path}")

    def load_recognizer(self):
        if os.path.exists(self.recog_path):
            prototypes = np.load(self.recog_path, allow_pickle=True).item()
            print(f"[MemoryCore] Loaded recognizer prototypes from {self.recog_path}")
            return prototypes
        else:
            print(f"[MemoryCore] No saved recognizer prototypes found at {self.recog_path}")
            return None
