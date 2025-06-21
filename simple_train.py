"""
simple_train.py
Train the model on a single letter (e.g., 'A') for quick mastery and memory save testing.
"""
import numpy as np
from src.core.advanced_trainer import AdvancedTrainer
from src.core.memory_core import MemoryCore
from src.core.rl_feedback import RLFeedback
from src.core.letter_image_generator import LetterImageGenerator
import string
from collections import defaultdict
import os

INPUT_SHAPE = (28, 28)
SDR_DIM = 256
LATENT_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2
HIDDEN_DIM = 64
LR = 0.05
MIN_LR = 0.001
MAX_LR = 0.2
ALPHA = 0.95
MOVING_AVG_N = 10
STAGNATION_STEPS = 5
MASTERY_THRESHOLD = 0.08
EPOCHS = 10000
LETTERS = list(string.digits)

print(f"[Config] Advanced mode: SDR_DIM={SDR_DIM}, LR={LR}, LETTERS={LETTERS}, EPOCHS={EPOCHS}")

trainer = AdvancedTrainer(INPUT_SHAPE, SDR_DIM, LATENT_DIM, NUM_LAYERS, NUM_HEADS, HIDDEN_DIM)
generator = LetterImageGenerator(*INPUT_SHAPE)
memory = MemoryCore('simple_decoder_weights.npy')
rl = RLFeedback(LETTERS, moving_avg_N=MOVING_AVG_N, stagnation_steps=STAGNATION_STEPS, reward_mode='strict')

# Track running average error for each letter
letter_errors = defaultdict(lambda: [])

# Try to load previous memory
loaded = memory.load()
if loaded is not None:
    # Optionally, load weights into hierarchical/advanced trainer if supported
    print(f"[MemoryCore] Loaded weights (integration with advanced trainer is modular)")

np.random.seed(42)
valence_history = []
mastered_letters = []

for letter in LETTERS:
    print(f"\n[Training] Starting mastery for letter: '{letter}'")
    letter_mastered = False
    epoch = 0
    letter_errors[letter] = []
    while not letter_mastered and epoch < EPOCHS:
        # Increase augmentation and difficulty for training images
        img = generator.get_image(letter, augment=True)
        result = trainer.train_step(img)
        error = np.abs(result['error']).mean()
        letter_errors[letter].append(error)
        action = None
        reward, lr_mod, _, _ = rl.update(letter, error, action, base_lr=LR, epoch=epoch)
        valence_history.append(rl.get_valence_ma(letter))
        print(f"Epoch {epoch+1} | Letter: {letter} | Error: {error:.4f} | Reward: {reward:+.4f} | ValenceMA: {rl.get_valence_ma(letter):.4f}")
        if (epoch + 1) % 100 == 0:
            mean_error = np.mean(letter_errors[letter])
            print(f"[Stats] Epoch {epoch+1}: Running mean error for '{letter}': {mean_error:.4f}")
        # Mastery: running mean error for this letter below threshold
        mean_error = np.mean(letter_errors[letter])
        if mean_error < MASTERY_THRESHOLD:
            letter_mastered = True
            mastered_letters.append(letter)
            print(f"[Mastery] Letter '{letter}' mastered in {epoch+1} epochs!")
            memory.save(None)  # Optionally, save advanced trainer state
        epoch += 1
    if not letter_mastered:
        print(f"[Warning] Did not master letter '{letter}' in allotted epochs.")

if len(mastered_letters) == len(LETTERS):
    print(f"All letters {LETTERS} mastered!")
else:
    print(f"Mastered letters: {mastered_letters}")
    print(f"Did not master all letters in allotted epochs.")
