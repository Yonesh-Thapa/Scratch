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
pattern = re.compile(r"Epoch (\d+) \| Letter: (\w) \| Prev: ([\d.NA-]+) \| Error: ([\d.]+) \| Reward: ([\-\d.]+) \| Plasticity: ([\d.eE+-]+) \| ValenceMA: ([\-\d.]+)")

def parse_log(filename):
    data = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            m = pattern.match(line)
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

def plot_metrics(data):
    letters = sorted(data.keys())
    epochs = [d['epoch'] for d in data[letters[0]]]
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    # Error
    for letter in letters:
        axs[0].plot(epochs, [d['error'] for d in data[letter]], label=letter)
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
    axs[3].set_xlabel('Epoch')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = parse_log(LOG_FILE)
    plot_metrics(data)
