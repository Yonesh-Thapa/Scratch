"""
rl_feedback.py
Implements RL-style feedback (reward/penalty) for letter learning.
Ultra-strict RL: strong negative rewards for stagnation/regression, forced exploration, action diversity penalties, adaptive plasticity, and enhanced logging.
"""
import numpy as np
from collections import deque, Counter
import random

class RLFeedback:
    def __init__(self, letters, moving_avg_N=20, stagnation_steps=10, exploration_steps=5, diversity_window=10, adaptive_lr=True, reward_mode='gentle', curriculum_phase=0):
        """
        Args:
            letters: list of all letters
            moving_avg_N: window for moving average valence
            stagnation_steps: steps to detect stagnation
            exploration_steps: steps of net negative reward before forced exploration
            diversity_window: window for action diversity tracking
            adaptive_lr: enable adaptive plasticity (learning rate)
            reward_mode: 'gentle' or 'strict' reward structure
            curriculum_phase: current phase of the curriculum (0: easy, 1: normal, 2: hard)
        """
        self.letters = letters
        self.prev_errors = {letter: None for letter in letters}
        self.valence = {letter: 0.0 for letter in letters}
        self.valence_ma = {letter: deque(maxlen=moving_avg_N) for letter in letters}
        self.global_valence_ma = deque(maxlen=moving_avg_N)
        self.moving_avg_N = moving_avg_N
        self.stagnation_steps = stagnation_steps
        self.exploration_steps = exploration_steps
        self.stagnation_counter = {letter: 0 for letter in letters}
        self.log = []
        self.reward_history = deque(maxlen=exploration_steps)
        self.forced_exploration = False
        self.diversity_window = diversity_window
        self.action_history = {letter: deque(maxlen=diversity_window) for letter in letters}
        self.diversity_scores = {letter: [] for letter in letters}
        self.penalty_counts = {'stagnation': 0, 'regression': 0, 'repetition': 0, 'forced_exploration': 0}
        self.negative_streak = {letter: 0 for letter in letters}
        self.adaptive_lr = adaptive_lr
        self.lr_mod = {letter: 1.0 for letter in letters}  # learning rate modifier
        self.lr_min = 0.1
        self.lr_max = 10.0
        self.lr_boost_steps = {letter: 0 for letter in letters}
        self.reward_mode = reward_mode  # 'gentle' or 'strict'
        self.curriculum_phase = {l: 0 for l in letters}  # per-letter phase
        self.low_error_streak = {l: 0 for l in letters}
        self.low_error_patience = 5  # N epochs of low error before increasing difficulty
        self.flat_error_streak = {l: 0 for l in letters}
        self.flat_error_patience = 8  # N epochs of flat error before forced exploration
        self.forced_exploration_streak = {l: 0 for l in self.letters}
        self.rollback_streak = {l: 0 for l in self.letters}
        self.curriculum_interventions = {l: 0 for l in self.letters}
        self.plasticity_boost_steps = {l: 0 for l in self.letters}
        self.diversity_bonus = {l: 0 for l in self.letters}
        self.replay_buffer = {l: [] for l in self.letters}
        self.best_error = {l: float('inf') for l in self.letters}
        self.best_action = {l: None for l in self.letters}
        self.negative_reward_streak = {l: 0 for l in self.letters}
        self.negative_streak_soften = 5
        self.last_errors = {l: [] for l in self.letters}  # for velocity
        self.velocity_window = 5
        self.recovery_cooldown = {l: 0 for l in self.letters}  # cooldown after recovery

    def update(self, letter, error, action=None, base_lr=None, epoch=None, rollback=False):
        """Update RL feedback for a letter and error, enforcing strict RL rules."""
        prev = self.prev_errors[letter]
        reward, event, recovery_triggered = self._compute_reward(letter, error, action, prev)
        diversity, velocity = self._update_diversity_velocity(letter, error, action)
        self._handle_penalties(letter, reward)
        self._track_streaks(letter, error, rollback)
        self._curriculum_and_plasticity(letter)
        self._update_replay_buffer(letter, error, action, epoch)
        self._update_valence(letter, reward)
        self._log_event(letter, error, reward, event, diversity, velocity, epoch, recovery_triggered, prev)
        forced_exploration = self._check_forced_exploration(letter, reward)
        self.prev_errors[letter] = error
        return reward, self.lr_mod[letter], self.flat_error_streak[letter] >= self.flat_error_patience, recovery_triggered

    def _compute_reward(self, letter, error, action, prev):
        # Biologically inspired, curiosity-driven, predictive, hierarchical, error-driven, novelty/attention-based reward
        reward = 0.0
        event = None
        recovery_triggered = False
        # --- Diversity encouragement (novelty/curiosity bonus) ---
        diversity = 1.0
        if action is not None:
            self.action_history[letter].append(action)
            unique = len(set(self.action_history[letter]))
            diversity = unique / max(1, len(self.action_history[letter]))
            self.diversity_scores[letter].append(diversity)
            if diversity > 0.8:  # Stronger bonus for high novelty
                reward += 1.0
                self.diversity_bonus[letter] += 2
                event = (event + '+novelty_bonus') if event else 'novelty_bonus'
            elif diversity > 0.5:
                reward += 0.5
                self.diversity_bonus[letter] += 1
                event = (event + '+diversity_bonus') if event else 'diversity_bonus'
        # --- Error reduction (prediction error minimization) ---
        velocity = 0.0
        if len(self.last_errors[letter]) >= 2:
            velocity = self.last_errors[letter][-2] - self.last_errors[letter][-1]
        # Strongly reward any reduction in error, even slight
        mastery_bonus = 0.0
        catastrophic_forgetting_penalty = 0.0
        sustained_low_error = False
        if prev is not None:
            # Catastrophic forgetting: punish if error increases on previously mastered letter
            if self.low_error_streak[letter] >= self.low_error_patience and error > prev + 0.05:
                catastrophic_forgetting_penalty = -2.0 * (error - prev)
                event = (event + '+catastrophic_forgetting') if event else 'catastrophic_forgetting'
            # Mastery bonus: sustained low error
            if error < 0.05:
                self.low_error_streak[letter] += 1
                if self.low_error_streak[letter] >= self.low_error_patience:
                    mastery_bonus = 1.5
                    sustained_low_error = True
                    event = (event + '+mastery') if event else 'mastery'
            else:
                self.low_error_streak[letter] = 0
        # --- Reward structure ---
        if self.reward_mode == 'strict':
            if prev is None:
                reward = 0.0
            elif error > prev:
                penalty = min(-1.5 - 2.5 * (error - prev), -0.3)
                reward = penalty + 0.25 * velocity + catastrophic_forgetting_penalty
                self.penalty_counts['regression'] += 1
                event = (event + '+regression') if event else 'regression'
                self.negative_streak[letter] += 1
            elif abs(error - prev) < 1e-6:
                reward = -1.2  # Increased penalty for stagnation
                self.penalty_counts['stagnation'] += 1
                event = (event + '+stagnation') if event else 'stagnation'
                self.negative_streak[letter] += 1
            else:
                # Strong reward for any error reduction
                reward = (prev - error) * 6.0 + 0.5 * velocity + mastery_bonus
                self.negative_streak[letter] = 0
        else:
            # gentle
            if prev is None:
                reward = 0.0
            else:
                reward = (prev - error) * 5.0 + 0.5 * velocity + mastery_bonus
                if abs(error - prev) < 1e-6:
                    reward = -0.3
                    self.flat_error_streak[letter] += 1
                else:
                    self.flat_error_streak[letter] = 0
                if error > prev:
                    reward = -0.7 * (error - prev) + catastrophic_forgetting_penalty
        # --- Penalty softening on negative streak ---
        if reward < 0:
            self.negative_reward_streak[letter] += 1
            if self.negative_reward_streak[letter] >= self.negative_streak_soften:
                reward *= 0.5
                if self.negative_reward_streak[letter] % (self.negative_streak_soften * 2) == 0:
                    reward += 0.2
        else:
            self.negative_reward_streak[letter] = 0
        # --- Plasticity boost for surprise/novelty/hard examples ---
        if (diversity > 0.8 or abs(prev - error) > 0.2 or (prev is not None and error > prev + 0.1)):
            self.lr_mod[letter] = min(self.lr_mod[letter] * 1.3, self.lr_max * 2)
            self.plasticity_boost_steps[letter] = 3
        # --- Exploration encouragement ---
        if diversity < 0.3 and not sustained_low_error:
            reward += 0.2  # Encourage exploration if not yet mastered
        return reward, event, recovery_triggered

    def _update_diversity_velocity(self, letter, error, action):
        diversity = 1.0
        if action is not None:
            self.action_history[letter].append(action)
            unique = len(set(self.action_history[letter]))
            diversity = unique / max(1, len(self.action_history[letter]))
            self.diversity_scores[letter].append(diversity)
        # Track error history for velocity
        self.last_errors[letter].append(error)
        if len(self.last_errors[letter]) > self.velocity_window:
            self.last_errors[letter].pop(0)
        velocity = 0.0
        if len(self.last_errors[letter]) >= 2:
            velocity = self.last_errors[letter][-2] - self.last_errors[letter][-1]
        return diversity, velocity

    def _handle_penalties(self, letter, reward):
        # --- Penalty softening on negative streak ---
        if reward < 0:
            self.negative_reward_streak[letter] += 1
            if self.negative_reward_streak[letter] >= self.negative_streak_soften:
                # Only soften penalty, do not handle event string here
                reward *= 0.5
                if self.negative_reward_streak[letter] % (self.negative_streak_soften * 2) == 0:
                    reward += 0.2
        else:
            self.negative_reward_streak[letter] = 0

    def _track_streaks(self, letter, error, rollback):
        # --- Forced exploration/rollback streak tracking ---
        if rollback:
            self.rollback_streak[letter] += 1
        else:
            self.rollback_streak[letter] = 0
        # Forced exploration streak
        if self.flat_error_streak[letter] >= self.flat_error_patience:
            self.forced_exploration_streak[letter] += 1
        else:
            self.forced_exploration_streak[letter] = 0
        # --- Curriculum intervention ---
        if self.forced_exploration_streak[letter] >= 3 or self.rollback_streak[letter] >= 3:
            # Reduce augmentation, log event (event handling moved to _compute_reward)
            self.curriculum_phase[letter] = 0
            self.curriculum_interventions[letter] += 1

    def _curriculum_and_plasticity(self, letter):
        # --- Plasticity boost ---
        if self.forced_exploration_streak[letter] >= 2:
            self.plasticity_boost_steps[letter] = 5
        if self.plasticity_boost_steps[letter] > 0:
            self.lr_mod[letter] = min(self.lr_mod[letter] * 1.5, self.lr_max * 2)
            self.plasticity_boost_steps[letter] -= 1

    def _update_replay_buffer(self, letter, error, action, epoch):
        # --- Replay buffer update ---
        if error < self.best_error[letter]:
            self.best_error[letter] = error
            self.best_action[letter] = action
            self.replay_buffer[letter].append((error, action, epoch))
            if len(self.replay_buffer[letter]) > 10:
                self.replay_buffer[letter].pop(0)

    def _update_valence(self, letter, reward):
        # Use a moving average of recent rewards instead of cumulative sum to bound valence
        if not hasattr(self, 'reward_ma'):
            self.reward_ma = {l: deque(maxlen=self.moving_avg_N) for l in self.letters}
        self.reward_ma[letter].append(reward)
        self.valence[letter] = np.mean(self.reward_ma[letter])
        self.valence_ma[letter].append(self.valence[letter])
        self.global_valence_ma.append(np.mean([self.valence[l] for l in self.letters]))
        # Stagnation detection
        prev = self.prev_errors[letter]
        error = self.valence[letter]  # Use current valence as a proxy for error if not passed
        if prev is not None and abs(error - prev) < 1e-6:
            self.stagnation_counter[letter] += 1
        else:
            self.stagnation_counter[letter] = 0
        self.prev_errors[letter] = error

    def _log_event(self, letter, error, reward, event, diversity, velocity, epoch, recovery_triggered, prev):
        # Enhanced log: epoch, letter, prev_error, new_error, reward, valence_ma, event, diversity, lr_mod
        entry = {
            'epoch': epoch,
            'letter': letter,
            'prev_error': prev,
            'error': error,
            'reward': reward,
            'valence': self.valence[letter],
            'valence_ma': np.mean(self.valence_ma[letter]),
            'global_valence_ma': np.mean(self.global_valence_ma),
            'stagnation': self.stagnation_counter[letter],
            'event': event,
            'diversity': diversity,
            'penalty_counts': self.penalty_counts.copy(),
            'forced_exploration': self.flat_error_streak[letter] >= self.flat_error_patience,
            'lr_mod': self.lr_mod[letter],
            'recovery_triggered': recovery_triggered,
            'curriculum_interventions': self.curriculum_interventions[letter],
            'diversity_bonus': self.diversity_bonus[letter],
            'velocity': velocity,
            'replay_buffer_size': len(self.replay_buffer[letter])
        }
        self.log.append(entry)

    def _check_forced_exploration(self, letter, reward):
        # Forced exploration if net reward negative for N steps
        self.reward_history.append(reward)
        forced_exploration = False
        if self.recovery_cooldown[letter] > 0:
            self.recovery_cooldown[letter] -= 1
        if sum(self.reward_history) < 0 and len(self.reward_history) == self.reward_history.maxlen and self.recovery_cooldown[letter] == 0:
            forced_exploration = True
            self.penalty_counts['forced_exploration'] += 1
            self.lr_boost_steps[letter] = 3  # boost learning rate for 3 steps
        # Recovery logic: if error increases for 3+ consecutive steps, trigger recovery
        if self.negative_streak[letter] >= 3:
            self.lr_mod[letter] = min(self.lr_mod[letter] * 2.5, self.lr_max * 2)
            self.negative_streak[letter] = 0
            self.recovery_cooldown[letter] = 3  # Set cooldown after recovery
        return forced_exploration

    def get_valence(self, letter):
        """Return current valence for a letter."""
        return self.valence[letter]

    def get_valence_ma(self, letter):
        """Return moving average valence for a letter."""
        return np.mean(self.valence_ma[letter])

    def get_global_valence_ma(self):
        """Return global moving average valence."""
        return np.mean(self.global_valence_ma)

    def get_log(self):
        """Return the full RL event log."""
        return self.log

    def get_penalty_counts(self):
        """Return penalty event counts."""
        return self.penalty_counts.copy()

    def get_diversity_score(self, letter):
        """Return most recent diversity score for a letter."""
        if self.diversity_scores[letter]:
            return self.diversity_scores[letter][-1]
        return 1.0

    def get_lr_mod(self, letter):
        """Return current learning rate modifier for a letter."""
        return self.lr_mod[letter]

    def set_reward_mode(self, mode):
        """Set the reward mode."""
        self.reward_mode = mode

    def set_curriculum_phase(self, phase):
        """Set the curriculum phase."""
        self.curriculum_phase = phase

    def get_curriculum_phase(self, letter):
        """Get the current curriculum phase."""
        return self.curriculum_phase[letter]

    def reset(self):
        """Reset all RL feedback state."""
        self.prev_errors = {letter: None for letter in self.letters}
        self.valence = {letter: 0.0 for letter in self.letters}
        self.valence_ma = {letter: deque(maxlen=self.moving_avg_N) for letter in self.letters}
        self.global_valence_ma = deque(maxlen=self.moving_avg_N)
        self.stagnation_counter = {letter: 0 for letter in self.letters}
        self.log = []
        self.reward_history.clear()
        self.forced_exploration = False
        self.action_history = {letter: deque(maxlen=self.diversity_window) for letter in self.letters}
        self.diversity_scores = {letter: [] for letter in self.letters}
        self.penalty_counts = {'stagnation': 0, 'regression': 0, 'repetition': 0, 'forced_exploration': 0}
        self.negative_streak = {letter: 0 for letter in self.letters}
        self.lr_mod = {letter: 1.0 for letter in self.letters}
        self.lr_boost_steps = {letter: 0 for letter in self.letters}
        self.forced_exploration_streak = {l: 0 for l in self.letters}
        self.rollback_streak = {l: 0 for l in self.letters}
        self.curriculum_interventions = {l: 0 for l in self.letters}
        self.plasticity_boost_steps = {l: 0 for l in self.letters}
        self.diversity_bonus = {l: 0 for l in self.letters}
        self.replay_buffer = {l: [] for l in self.letters}
        self.best_error = {l: float('inf') for l in self.letters}
        self.best_action = {l: None for l in self.letters}
        self.negative_reward_streak = {l: 0 for l in self.letters}
        self.last_errors = {l: [] for l in self.letters}  # for velocity
