"""
rl_log_visualizer.py
Visualizes RL training logs: error, reward, plasticity, and moving average valence over epochs.
Upgraded: parses enhanced RL log format.
"""
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

LOG_FILE = "rl_feedback_log.txt"
MOVING_AVG_WINDOW = 20

# Parse the enhanced RL log
# Expected format: Epoch N | Letter: X | Prev: ... | Error: ... | Reward: ... | Plasticity: ... | ValenceMA: ...
pattern = re.compile(r"Epoch (\d+) \| Letter: (\w) \| Prev: ([^|]+) \| Error: ([^|]+) \| Reward: ([^|]+) \| Plasticity: ([^|]+) \| ValenceMA: ([^|]+) \| Event: ([^|]+) \| Diversity: ([^|]+) \| Penalties: (\{[^}]+\}) \| ForcedExploration: ([^|]+) \| LRMod: ([^|]+)")

def parse_log(filename):
    # Enhanced parser: supports both old and new log formats
    data = defaultdict(list)
    # Try enhanced pattern first
    enhanced_pattern = re.compile(r"Epoch (\d+) \| Letter: (\w) \| Prev: ([^|]+) \| Error: ([^|]+) \| Reward: ([^|]+) \| Plasticity: ([^|]+) \| ValenceMA: ([^|]+) \| Event: ([^|]+) \| Diversity: ([^|]+) \| Penalties: (\{[^}]+\}) \| ForcedExploration: ([^|]+) \| LRMod: ([^|]+)")
    basic_pattern = re.compile(r"Epoch (\d+) \| Letter: (\w) \| Prev: ([\d.NA-]+) \| Error: ([\d.]+) \| Reward: ([\-\d.]+) \| Plasticity: ([\d.eE+-]+) \| ValenceMA: ([\-\d.]+)")
    with open(filename, "r") as f:
        for line in f:
            m = enhanced_pattern.match(line)
            if m:
                epoch, letter, prev, err, reward, plasticity, val_ma, event, diversity, penalties, forced, lrmod = m.groups()
                data[letter].append({
                    'epoch': int(epoch),
                    'prev_error': float(prev) if prev != 'NA' else None,
                    'error': float(err),
                    'reward': float(reward),
                    'plasticity': float(plasticity),
                    'valence_ma': float(val_ma),
                    'event': event,
                    'diversity': float(diversity),
                    'penalties': penalties,
                    'forced_exploration': forced == 'True',
                    'lr_mod': float(lrmod)
                })
            else:
                m = basic_pattern.match(line)
                if m:
                    epoch, letter, prev, err, reward, plasticity, val_ma = m.groups()
                    data[letter].append({
                        'epoch': int(epoch),
                        'prev_error': float(prev) if prev != 'NA' else None,
                        'error': float(err),
                        'reward': float(reward),
                        'plasticity': float(plasticity),
                        'valence_ma': float(val_ma)
                    })
    return data

def plot_metrics(data, save_path=None, show=True, highlight_drops=True):
    letters = sorted(data.keys())
    epochs = [d['epoch'] for d in data[letters[0]]]
    fig, axs = plt.subplots(8, 1, figsize=(14, 32), sharex=True)
    # Error
    for letter in letters:
        errors = [d['error'] for d in data[letter]]
        axs[0].plot(epochs, errors, label=letter)
        if highlight_drops:
            drops = [i for i in range(1, len(errors)) if errors[i] - errors[i-1] > 0.05]
            axs[0].scatter([epochs[i] for i in drops], [errors[i] for i in drops], color='red', s=40, marker='v', label=f'{letter} drop')
    axs[0].set_ylabel('Error')
    axs[0].legend()
    axs[0].set_title('Error over Epochs')
    # Reward
    for letter in letters:
        axs[1].plot(epochs, [d['reward'] for d in data[letter]], label=letter)
    axs[1].set_ylabel('Reward')
    axs[1].set_title('Reward over Epochs')
    # Plasticity
    for letter in letters:
        axs[2].plot(epochs, [d['plasticity'] for d in data[letter]], label=letter)
    axs[2].set_ylabel('Plasticity')
    axs[2].set_title('Plasticity over Epochs')
    # Moving average valence
    for letter in letters:
        axs[3].plot(epochs, [d['valence_ma'] for d in data[letter]], label=letter)
    axs[3].set_ylabel('Valence (Moving Avg)')
    axs[3].set_title('Moving Average Valence over Epochs')
    # Learning rate (if present)
    if any('lr_mod' in d for d in data[letters[0]]):
        for letter in letters:
            axs[4].plot(epochs, [d.get('lr_mod', np.nan) for d in data[letter]], label=letter)
        axs[4].set_ylabel('Learning Rate')
        axs[4].set_title('Learning Rate over Epochs')
    # Error rate (change per epoch)
    for letter in letters:
        errors = [d['error'] for d in data[letter]]
        error_rate = [errors[i] - errors[i-1] for i in range(1, len(errors))]
        axs[5].plot(epochs[1:], error_rate, label=letter)
    axs[5].set_ylabel('Error Rate (ΔError)')
    axs[5].set_title('Error Rate (Change per Epoch)')
    # Improvement rate (negative error rate)
    for letter in letters:
        errors = [d['error'] for d in data[letter]]
        improvement_rate = [-(errors[i] - errors[i-1]) for i in range(1, len(errors))]
        axs[6].plot(epochs[1:], improvement_rate, label=letter)
    axs[6].set_ylabel('Improvement Rate')
    axs[6].set_title('Improvement Rate (Negative ΔError)')
    # Converging/diverging indicator (rolling std of error)
    window = 5
    for letter in letters:
        errors = [d['error'] for d in data[letter]]
        rolling_std = [np.std(errors[max(0, i-window+1):i+1]) for i in range(len(errors))]
        axs[7].plot(epochs, rolling_std, label=letter)
    axs[7].set_ylabel('Error Std (Converge/Diverge)')
    axs[7].set_title('Converging/Diverging (Rolling Error Std)')
    axs[7].set_xlabel('Epoch')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="RL Log Visualizer: Plot error, reward, plasticity, valence, and more.")
    parser.add_argument('--log', type=str, default=LOG_FILE, help='Path to RL log file')
    parser.add_argument('--save', type=str, default=None, help='Path to save plot image (PNG)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot interactively')
    parser.add_argument('--no-highlight', action='store_true', help='Do not highlight drops/improvements')
    parser.add_argument('--live', action='store_true', help='Enable live mode: update plot as log file changes')
    parser.add_argument('--interval', type=float, default=2.0, help='Refresh interval (seconds) for live mode')
    args = parser.parse_args()

    if args.live:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axs = plt.subplots(8, 1, figsize=(14, 32), sharex=True)
        # Persistent line objects for each metric/letter
        line_objs = [{} for _ in range(8)]  # 8 axes
        last_letters = set()
        window = 5
        while True:
            try:
                data = parse_log(args.log)
                if not data:
                    time.sleep(args.interval)
                    continue
                letters = sorted(data.keys())
                epochs = [d['epoch'] for d in data[letters[0]]]
                # If letters changed, clear axes and reset lines
                if set(letters) != last_letters:
                    for ax in axs:
                        ax.clear()
                    line_objs = [{} for _ in range(8)]
                    last_letters = set(letters)
                # --- Update or create lines for each metric ---
                for idx, metric in enumerate(['error', 'reward', 'plasticity', 'valence_ma', 'lr_mod', 'error_rate', 'improvement_rate', 'rolling_std']):
                    for letter in letters:
                        # Prepare y-data for each metric
                        if metric == 'error':
                            y = [d['error'] for d in data[letter]]
                        elif metric == 'reward':
                            y = [d['reward'] for d in data[letter]]
                        elif metric == 'plasticity':
                            y = [d['plasticity'] for d in data[letter]]
                        elif metric == 'valence_ma':
                            y = [d['valence_ma'] for d in data[letter]]
                        elif metric == 'lr_mod':
                            y = [d.get('lr_mod', np.nan) for d in data[letter]]
                            if not any(np.isfinite(y)):
                                continue  # skip if no lr_mod
                        elif metric == 'error_rate':
                            errors = [d['error'] for d in data[letter]]
                            y = [errors[i] - errors[i-1] for i in range(1, len(errors))]
                        elif metric == 'improvement_rate':
                            errors = [d['error'] for d in data[letter]]
                            y = [-(errors[i] - errors[i-1]) for i in range(1, len(errors))]
                        elif metric == 'rolling_std':
                            errors = [d['error'] for d in data[letter]]
                            y = [np.std(errors[max(0, i-window+1):i+1]) for i in range(len(errors))]
                        # x-data
                        if metric in ['error_rate', 'improvement_rate']:
                            x = epochs[1:]
                        else:
                            x = epochs
                        # Update or create line
                        if letter in line_objs[idx]:
                            line_objs[idx][letter].set_data(x, y)
                        else:
                            (l,) = axs[idx].plot(x, y, label=letter)
                            line_objs[idx][letter] = l
                # Remove lines for letters no longer present
                for idx in range(8):
                    for old_letter in list(line_objs[idx].keys()):
                        if old_letter not in letters:
                            line_objs[idx][old_letter].remove()
                            del line_objs[idx][old_letter]
                # --- Highlight drops for error plot ---
                if not args.no_highlight:
                    for coll in list(axs[0].collections):
                        coll.remove()
                    for letter in letters:
                        errors = [d['error'] for d in data[letter]]
                        drops = [i for i in range(1, len(errors)) if errors[i] - errors[i-1] > 0.05]
                        axs[0].scatter([epochs[i] for i in drops], [errors[i] for i in drops], color='red', s=40, marker='v', label=f'{letter} drop')
                # --- Set labels/titles ---
                axs[0].set_ylabel('Error')
                axs[0].set_title('Error over Epochs')
                axs[1].set_ylabel('Reward')
                axs[1].set_title('Reward over Epochs')
                axs[2].set_ylabel('Plasticity')
                axs[2].set_title('Plasticity over Epochs')
                axs[3].set_ylabel('Valence (Moving Avg)')
                axs[3].set_title('Moving Average Valence over Epochs')
                if any('lr_mod' in d for d in data[letters[0]]):
                    axs[4].set_ylabel('Learning Rate')
                    axs[4].set_title('Learning Rate over Epochs')
                else:
                    axs[4].set_ylabel('Learning Rate')
                    axs[4].set_title('Learning Rate (N/A)')
                axs[5].set_ylabel('Error Rate (ΔError)')
                axs[5].set_title('Error Rate (Change per Epoch)')
                axs[6].set_ylabel('Improvement Rate')
                axs[6].set_title('Improvement Rate (Negative ΔError)')
                axs[7].set_ylabel('Error Std (Converge/Diverge)')
                axs[7].set_title('Converging/Diverging (Rolling Error Std)')
                axs[7].set_xlabel('Epoch')
                # --- Legends ---
                for ax in axs:
                    handles, labels_ = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend()
                plt.tight_layout()
                fig.canvas.draw_idle()
                plt.pause(0.01)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("[Live Visualizer] Interrupted by user.")
                break
    else:
        data = parse_log(args.log)
        plot_metrics(data, save_path=args.save, show=not args.no_show, highlight_drops=not args.no_highlight)
# USAGE:
# python core/rl_log_visualizer.py --log rl_feedback_log.txt --save out.png
