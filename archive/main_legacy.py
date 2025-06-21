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
from core.reconstruction_trainer import ReconstructionTrainer
from core.brain_visualizer import BrainVisualizer
from core.letter_image_generator import LetterImageGenerator
from core.memory_core import MemoryCore
from core.rl_feedback import RLFeedback

# Parameters
INPUT_SHAPE = (28, 28)
SDR_DIM = 512
SPARSITY = 0.05
LR = 0.01
MASTERY_THRESHOLD = 0.1  # Stop when all letters have error below this
MAX_EPOCHS = 10000
ALPHA = 0.1  # Plasticity modulation factor
MIN_LR = 1e-5
MAX_LR = 0.1
MOVING_AVG_N = 20
STAGNATION_STEPS = 10
FORCED_EXPLORATION_STEPS = 5  # Steps of net negative reward before forced exploration
N_CONSECUTIVE_NEGATIVE = 5

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

# Try to load previous memory
loaded = memory.load()
if loaded is not None:
    trainer.decoder.W = loaded

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

while not mastered and epoch < MAX_EPOCHS:
    errors = []
    rewards = []
    diversity_scores = []
    penalty_counts = None
    for letter in letters:
        # Curriculum: set augmentation based on RL phase
        phase = rl.get_curriculum_phase(letter)
        if args.debug or phase == 0:
            generator.set_minimal_augmentation(letter)
        elif phase == 1:
            generator.set_medium_augmentation(letter)
        else:
            generator.aug_params[letter] = generator.default_aug_params.copy()
        # Choose augmentation/action id for diversity tracking
        img, action_id = generator.get_image(letter, augment=True, return_action=True)
        # Convert action_id dict to a hashable tuple for RL diversity
        if isinstance(action_id, dict):
            action_id = tuple(sorted(action_id.items()))
        result = trainer.train_step(img, letter=letter)
        valence = 1.0 - np.mean(np.abs(result['error']))
        valence_history.append(valence)
        error = np.mean(np.abs(result['error']))
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
        # Adaptive plasticity: set per-letter learning rate
        trainer.set_learning_rate(letter, trainer.get_base_lr(letter) * lr_mod)
        # Forced exploration/curriculum shift
        if forced_exploration:
            print(f"[StrictRL] Forced exploration triggered for letter {letter}!")
            generator.increase_augmentation(letter)
            trainer.set_learning_rate(letter, trainer.get_base_lr(letter) * lr_mod * 2.0)
        # Recovery logic: if recovery_triggered, reduce augmentation and boost plasticity
        if recovery_triggered:
            print(f"[RLRecovery] Recovery triggered for letter {letter}!")
            generator.reduce_augmentation(letter)
            trainer.set_learning_rate(letter, min(trainer.get_base_lr(letter) * lr_mod * 2.5, MAX_LR * 2))
        visualizer.update(
            result['input'],
            result['reconstruction'],
            result['error'],
            result['sdr'],
            result['attn_map'],
            result['context'],
            label=letter
        )
        print(f"Epoch {epoch+1} | Letter: {letter} | Prev: {prev_error if prev_error is not None else 'NA'} | Error: {error:.4f} | Reward: {reward:+.4f} | Plasticity: {trainer.get_learning_rate(letter):.5f} | ValenceMA: {rl.get_valence_ma(letter):.4f} | Stag: {rl.stagnation_counter[letter]} | Diversity: {diversity_scores[-1]:.2f} | Event: {rl.log[-1]['event']}")
        # After training step, decode the SDR to get the model's guess
        recon = trainer.decoder.decode(result['sdr']).reshape(INPUT_SHAPE)
        recognized = letter  # Placeholder: replace with classifier if available
        print(f"Given: {letter} | Recognized: {recognized}")
        time.sleep(0.01)
        # Save best weights for rollback
        if error < best_errors[letter]:
            best_errors[letter] = error
            best_weights[letter] = copy.deepcopy(trainer.decoder.W)
        # Rollback if error explodes
        if error > best_errors[letter] * 2.5 and best_weights[letter] is not None:
            print(f"[Rollback] Rolling back {letter} to best state!")
            trainer.decoder.W = copy.deepcopy(best_weights[letter])
            error = best_errors[letter]
    # Elasticity: grow if needed
    if errors and max(errors) > grow_threshold:
        print("[Elasticity] Growing neurons!")
        trainer.grow_neurons(max_growth)
        visualizer.sdr_dim = trainer.sdr_dim
        visualizer.im_sdr.set_data(np.zeros((1, trainer.sdr_dim)))
    # Prune rarely used neurons
    if errors and epoch % prune_interval == 0 and epoch > 0:
        print("[Elasticity] Pruning inactive neurons!")
        trainer.prune_neurons(min_activity=2)
        visualizer.sdr_dim = trainer.sdr_dim
        visualizer.im_sdr.set_data(np.zeros((1, trainer.sdr_dim)))
    # Rewire underperforming neurons
    if errors and epoch % rewire_interval == 0 and epoch > 0:
        underperforming = np.where(trainer.sdr_activity < 2)[0]
        if len(underperforming) > 0:
            print(f"[Elasticity] Rewiring {len(underperforming)} neurons!")
            trainer.rewire_neurons(underperforming)
    epoch += 1
    if errors and all(e < MASTERY_THRESHOLD for e in errors):
        mastered = True
        print(f"All letters mastered! Saving learned patterns...")
        memory.save(trainer.decoder.W)
    # Stay in gentle reward mode until all errors < 0.2 for 10 epochs
    if not args.no_rl and rl.reward_mode == 'gentle' and epoch > 10 and errors and all(e < 0.2 for e in errors):
        rl.set_reward_mode('strict')
        print('[RL] Switching to strict reward mode!')

# Enhanced RL log: epoch, letter, prev_error, new_error, reward, plasticity, moving average valence, event, diversity, penalty counts
with open("rl_feedback_log.txt", "w") as f:
    for i, entry in enumerate(rl.get_log()):
        f.write(f"Epoch {i+1} | Letter: {entry['letter']} | Prev: {entry['prev_error'] if entry['prev_error'] is not None else 'NA'} | Error: {entry['error']:.4f} | Reward: {entry['reward']:+.4f} | Plasticity: {trainer.get_learning_rate(entry['letter']):.5f} | ValenceMA: {entry['valence_ma']:.4f} | Event: {entry['event']} | Diversity: {entry['diversity']:.2f} | Penalties: {entry['penalty_counts']} | ForcedExploration: {entry['forced_exploration']} | LRMod: {entry['lr_mod']:.3f}\n")

print("Training complete. Close the matplotlib window to exit.")
input("Press Enter to exit...")
