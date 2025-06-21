import numpy as np
import os
import sys
sys.path.append('src')
from core.letter_image_generator import LetterImageGenerator
from core.reconstruction_trainer import ReconstructionTrainer
from core.recognizer import Recognizer

# Paths to memory files
RECOGNIZER_PATH = 'recognizer_prototypes.npy'
DECODER_PATH = 'simple_decoder_weights.npy'
ATTN_PATH = 'attention_weights.npy'

# --- Parameters (should match main.py) ---
INPUT_SHAPE = (28, 28)
SDR_DIM = 512
SPARSITY = 0.05
LR = 0.01
MASTERY_THRESHOLD = 0.1

def print_recognizer_prototypes():
    if not os.path.exists(RECOGNIZER_PATH):
        print('No recognizer prototypes found.')
        return
    d = np.load(RECOGNIZER_PATH, allow_pickle=True).item()
    print(f"Loaded {len(d)} prototypes:")
    for k, v in d.items():
        print(f"  '{k}': shape={v.shape}, sample={v[:5] if hasattr(v, 'shape') else v}")

def print_decoder_weights():
    if not os.path.exists(DECODER_PATH):
        print('No decoder weights found.')
        return
    arr = np.load(DECODER_PATH, allow_pickle=True)
    print(f"Decoder weights shape: {arr.shape}")
    print(f"Sample weights (first row): {arr[0][:10]}")

def print_attention_weights():
    if not os.path.exists(ATTN_PATH):
        print('No attention weights found.')
        return
    d = np.load(ATTN_PATH, allow_pickle=True).item()
    print(f"Attention W shape: {d['W'].shape}")
    print(f"Attention b shape: {d['b'].shape}")
    print(f"Sample W (first head, first 5): {d['W'][0].flatten()[:5]}")

def print_mastered_letters():
    print('\n--- Mastered Letters ---')
    generator = LetterImageGenerator(*INPUT_SHAPE)
    letters = generator.letters
    trainer = ReconstructionTrainer(INPUT_SHAPE, SDR_DIM, SPARSITY, LR)
    recognizer = Recognizer(letters, trainer.hierarchical.latent_dim)
    # Load prototypes and decoder
    import os
    if os.path.exists(RECOGNIZER_PATH):
        d = np.load(RECOGNIZER_PATH, allow_pickle=True).item()
        recognizer.prototypes = d
    if os.path.exists(DECODER_PATH):
        arr = np.load(DECODER_PATH, allow_pickle=True)
        trainer.hierarchical.decoder = arr
    mastered_letters = []
    for letter in letters:
        img = generator.get_image(letter, augment=False)
        attended, attn_map = trainer.attention.apply_attention(img)
        # Robust flattening: use attended if shape is (28,28), else fallback to img
        if isinstance(attended, np.ndarray) and attended.shape == (28, 28):
            flat_attended = attended.flatten()
        elif isinstance(img, np.ndarray) and img.shape == (28, 28):
            flat_attended = img.flatten()
        else:
            # Fallback: flatten whatever is available
            flat_attended = np.array(attended).flatten()
        zs, _ = trainer.hierarchical.encode(flat_attended)
        sdr = zs[-1]
        pred_label, dist = recognizer.recognize(sdr)
        error = np.linalg.norm(sdr - recognizer.prototypes[letter])
        if error < MASTERY_THRESHOLD:
            mastered_letters.append(letter)
    print(f"Mastered letters ({len(mastered_letters)}): {sorted(mastered_letters)}")

def main():
    print('--- Recognizer Prototypes ---')
    print_recognizer_prototypes()
    print('\n--- Decoder Weights ---')
    print_decoder_weights()
    print('\n--- Attention Weights ---')
    print_attention_weights()
    print_mastered_letters()

if __name__ == '__main__':
    main()
