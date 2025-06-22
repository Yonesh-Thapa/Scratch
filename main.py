"""
main.py
Entry point: runs the full training and visualization loop with mastery-based stopping, valence, and memory core.
Upgraded: RL reward scaling, proportional plasticity, moving average valence, stagnation detection, enhanced RL log, and documentation.
"""
import numpy as np
import time
import random
import argparse
import copy
import sys
from src.core.reconstruction_trainer import ReconstructionTrainer
from src.core.brain_visualizer import BrainVisualizer
from src.core.letter_image_generator import LetterImageGenerator
from src.core.memory_core import MemoryCore
from src.core.rl_feedback import RLFeedback
from src.core.recognizer import Recognizer

# Parameters
INPUT_SHAPE = (28, 28)
SDR_DIM = 512
SPARSITY = 0.05
LR = 0.05           # Higher initial learning rate
MASTERY_THRESHOLD = 0.1
MAX_EPOCHS = 10000
ALPHA = 0.1
MIN_LR = 1e-4       # Higher min LR
MAX_LR = 0.1
MOVING_AVG_N = 20
STAGNATION_STEPS = 10
FORCED_EXPLORATION_STEPS = 5
N_CONSECUTIVE_NEGATIVE = 5
EASY_EPOCHS = 20    # Number of epochs with easy mode

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--no-rl', action='store_true', help='Run in supervised-only mode (no RL)')
parser.add_argument('--gentle-rl', action='store_true', help='Use gentle RL reward mode')
parser.add_argument('--easy', action='store_true', help='Start with easy curriculum (minimal augmentation)')
parser.add_argument('--letters', type=int, default=None, help='Limit to N letters for debugging')
parser.add_argument('--debug', action='store_true', help='Debug mode: 3 letters, no augmentation')
args = parser.parse_args()

# Create modules
trainer = ReconstructionTrainer(INPUT_SHAPE, SDR_DIM, SPARSITY, LR, min_lr=MIN_LR, max_lr=MAX_LR, alpha=ALPHA)
visualizer = BrainVisualizer(INPUT_SHAPE, SDR_DIM)
generator = LetterImageGenerator(*INPUT_SHAPE)
letters = generator.letters
memory = MemoryCore()
rl = RLFeedback(letters, moving_avg_N=MOVING_AVG_N, stagnation_steps=STAGNATION_STEPS, reward_mode='gentle' if args.gentle_rl else 'strict')
recognizer = Recognizer(letters, trainer.hierarchical.latent_dim)

# Try to load previous memory
loaded = memory.load_decoder()
if loaded is not None:
    trainer.hierarchical.decoder = loaded
# Load attention weights if available
W, b = memory.load_attention()
if W is not None and b is not None:
    trainer.attention.W = W
    trainer.attention.b = b
# Load recognizer prototypes if available
protos = memory.load_recognizer()
if protos is not None:
    recognizer.prototypes = protos

np.random.seed(42)
mastered = False
valence_history = []
epoch = 0
prune_interval = 10
rewire_interval = 20
grow_threshold = 0.25  # If any letter error > this, grow neurons
max_growth = 64  # Max neurons to add at once
curriculum_easy_mode = False
consecutive_negative = 0

if args.letters:
    letters = letters[:args.letters]

curriculum_easy_mode = args.easy

if args.debug:
    letters = letters[:3]
    for l in letters:
        generator.set_minimal_augmentation(l)

# Track best weights per letter for rollback
best_errors = {l: float('inf') for l in letters}
best_weights = {l: None for l in letters}
mastered_letters = set()

try:
    while not mastered:
        errors = []
        rewards = []
        diversity_scores = []
        penalty_counts = None
        # EASY MODE for first N epochs
        easy_mode = epoch < EASY_EPOCHS
        for letter in letters:
            # Curriculum: set augmentation based on RL phase or easy mode
            if args.debug or easy_mode or rl.get_curriculum_phase(letter) == 0:
                generator.set_minimal_augmentation(letter)
            elif rl.get_curriculum_phase(letter) == 1:
                generator.set_medium_augmentation(letter)
            else:
                generator.aug_params[letter] = generator.default_aug_params.copy()
            img, action_id = generator.get_image(letter, augment=True, return_action=True)
            if isinstance(action_id, dict):
                action_id = tuple(sorted(action_id.items()))
            result = trainer.train_step(img, letter=letter)
            recognizer.update(letter, result['zs'][-1])
            valence = 1.0 - np.mean(np.abs(result['error']))
            valence_history.append(valence)
            error = np.mean(np.abs(result['error']))
            if error < MASTERY_THRESHOLD:
                mastered_letters.add(letter)
            else:
                mastered_letters.discard(letter)
            prev_error = rl.prev_errors[letter]
            if args.no_rl:
                reward = 0.0
                lr_mod = 1.0
                forced_exploration = False
                recovery_triggered = False
            else:
                reward, lr_mod, forced_exploration, recovery_triggered = rl.update(letter, error, action=action_id, base_lr=trainer.get_base_lr(letter), epoch=epoch)
            rewards.append(reward)
            diversity_scores.append(rl.get_diversity_score(letter))
            penalty_counts = rl.get_penalty_counts()
            trainer.set_learning_rate(letter, trainer.get_base_lr(letter) * lr_mod)
            if forced_exploration:
                print(f"[StrictRL] Forced exploration triggered for letter {letter}!")
                generator.increase_augmentation(letter)
                trainer.set_learning_rate(letter, trainer.get_base_lr(letter) * lr_mod * 2.0)
            if recovery_triggered:
                print(f"[RLRecovery] Recovery triggered for letter {letter}!")
                generator.reduce_augmentation(letter)
                trainer.set_learning_rate(letter, min(trainer.get_base_lr(letter) * lr_mod * 2.5, MAX_LR * 2))
            trainer.visualize_step(visualizer, result, label=letter)
            print(f"Epoch {epoch+1} | Letter: {letter} | Prev: {prev_error if prev_error is not None else 'NA'} | Error: {error:.4f} | Reward: {reward:+.4f} | Plasticity: {trainer.get_learning_rate(letter):.5f} | ValenceMA: {rl.get_valence_ma(letter):.4f} | Stag: {rl.stagnation_counter[letter]} | Diversity: {diversity_scores[-1]:.2f} | Event: {rl.log[-1]['event']}")
            # Save best weights for rollback (and only save to disk if improved!)
            if error < best_errors[letter]:
                best_errors[letter] = error
                best_weights[letter] = copy.deepcopy(trainer.hierarchical.decoder)
                memory.save_decoder(trainer.hierarchical.decoder)
                memory.save_attention(trainer.attention.W, trainer.attention.b)
                memory.save_recognizer(recognizer.prototypes)
            # Remove save/load from recognition to avoid unnecessary I/O
            if error > best_errors[letter] * 2.5 and best_weights[letter] is not None:
                print(f"[Rollback] Rolling back {letter} to best state!")
                trainer.hierarchical.decoder = copy.deepcopy(best_weights[letter])
                error = best_errors[letter]
        if errors and max(errors) > grow_threshold:
            print("[Elasticity] Growing neurons!")
            trainer.grow_neurons(max_growth)
            visualizer.sdr_dim = trainer.sdr_dim
            visualizer.im_sdr.set_data(np.zeros((1, trainer.sdr_dim)))
        if errors and epoch % prune_interval == 0 and epoch > 0:
            print("[Elasticity] Pruning inactive neurons!")
            trainer.prune_neurons(min_activity=2)
            visualizer.sdr_dim = trainer.sdr_dim
            visualizer.im_sdr.set_data(np.zeros((1, trainer.sdr_dim)))
        if errors and epoch % rewire_interval == 0 and epoch > 0:
            underperforming = np.where(trainer.sdr_activity < 2)[0]
            if len(underperforming) > 0:
                print(f"[Elasticity] Rewiring {len(underperforming)} neurons!")
                trainer.rewire_neurons(underperforming)
        epoch += 1
        if errors and all(e < MASTERY_THRESHOLD for e in errors):
            mastered = True
            print(f"All letters mastered! Saving learned patterns...")
            memory.save(trainer.hierarchical.decoder)
        if not args.no_rl and rl.reward_mode == 'gentle' and epoch > 10 and errors and all(e < 0.2 for e in errors):
            rl.set_reward_mode('strict')
            print('[RL] Switching to strict reward mode!')
        print(f"Mastered letters so far: {sorted(mastered_letters)}")
except KeyboardInterrupt:
    print("\n[Exit] Training interrupted by user. Progress saved. Have a great day!")
    for letter in letters:
        if best_weights[letter] is not None:
            memory.save_decoder(best_weights[letter])
    sys.exit(0)

with open("rl_feedback_log.txt", "w") as f:
    for i, entry in enumerate(rl.get_log()):
        f.write(f"Epoch {i+1} | Letter: {entry['letter']} | Prev: {entry['prev_error'] if entry['prev_error'] is not None else 'NA'} | Error: {entry['error']:.4f} | Reward: {entry['reward']:+.4f} | Plasticity: {trainer.get_learning_rate(entry['letter']):.5f} | ValenceMA: {entry['valence_ma']:.4f} | Event: {entry['event']} | Diversity: {entry['diversity']:.2f} | Penalties: {entry['penalty_counts']} | ForcedExploration: {entry['forced_exploration']} | LRMod: {entry['lr_mod']:.3f}\n")

print("Training complete. Close the matplotlib window to exit.")

print("\nRecognition/classification test:")
correct = 0
n_test = 0
for letter in letters:
    img, _ = generator.get_image(letter, augment=False)
    features = trainer.feature_extractor.extract(img)
    attended, attn_map = trainer.attention.apply_attention(img)
    flat_attended = attended.flatten() if hasattr(attended, 'flatten') else attended
    zs, _ = trainer.hierarchical.encode(flat_attended)
    sdr = zs[-1]
    pred_label, dist = recognizer.recognize(sdr)
    print(f"True: {letter} | Predicted: {pred_label} | Distance: {dist:.4f}")
    if pred_label == letter:
        correct += 1
    n_test += 1
print(f"Accuracy: {correct}/{n_test} ({100.0 * correct / n_test:.2f}%)")
input("Press Enter to exit...")
