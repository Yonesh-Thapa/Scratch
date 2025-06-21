"""
rl_log_visualizer.py
Visualize RL log: error, reward, plasticity, valence, penalty counts, diversity, and forced exploration events.
"""
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_log(log_path):
    # Parse the RL log file into a DataFrame
    pattern = re.compile(r"Epoch (\d+) \| Letter: (\w) \| Prev: ([^|]+) \| Error: ([^|]+) \| Reward: ([^|]+) \| Plasticity: ([^|]+) \| ValenceMA: ([^|]+) \| Event: ([^|]+) \| Diversity: ([^|]+) \| Penalties: (\{[^}]+\}) \| ForcedExploration: ([^|]+) \| LRMod: ([^|]+)")
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                epoch, letter, prev, error, reward, plasticity, valence_ma, event, diversity, penalties, forced, lrmod = m.groups()
                records.append({
                    'epoch': int(epoch),
                    'letter': letter,
                    'error': float(error),
                    'reward': float(reward),
                    'plasticity': float(plasticity),
                    'valence_ma': float(valence_ma),
                    'event': event,
                    'diversity': float(diversity),
                    'penalties': penalties,
                    'forced_exploration': forced == 'True',
                    'lr_mod': float(lrmod)
                })
    return pd.DataFrame(records)

def plot_rl_log(log_path, baseline_path=None):
    df = parse_log(log_path)
    letters = sorted(df['letter'].unique())
    epochs = sorted(df['epoch'].unique())
    fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    for letter in letters[:5]:
        ldf = df[df['letter'] == letter]
        axs[0].plot(ldf['epoch'], ldf['error'], label=letter)
        axs[1].plot(ldf['epoch'], ldf['reward'], label=letter)
        axs[2].plot(ldf['epoch'], ldf['valence_ma'], label=letter)
        axs[3].plot(ldf['epoch'], ldf['diversity'], label=letter)
        # Curriculum interventions
        if 'curriculum_interventions' in ldf:
            interventions = ldf[ldf['curriculum_interventions'] > 0]
            axs[0].scatter(interventions['epoch'], interventions['error'], color='orange', s=30, marker='x', label=f'{letter} curriculum')
        # Diversity bonuses
        if 'diversity_bonus' in ldf:
            diversity_events = ldf[ldf['diversity_bonus'] > 0]
            axs[3].scatter(diversity_events['epoch'], diversity_events['diversity'], color='purple', s=30, marker='*', label=f'{letter} diversity')
    axs[0].set_ylabel("Error")
    axs[1].set_ylabel("Reward")
    axs[2].set_ylabel("ValenceMA")
    axs[3].set_ylabel("Diversity")
    axs[3].set_xlabel("Epoch")
    for ax in axs:
        ax.legend()
    # If baseline provided, plot for comparison
    if baseline_path:
        bdf = parse_log(baseline_path)
        for letter in letters[:5]:
            ldf = bdf[bdf['letter'] == letter]
            axs[0].plot(ldf['epoch'], ldf['error'], '--', label=f'{letter} (baseline)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        plot_rl_log(sys.argv[1], sys.argv[2])
    else:
        plot_rl_log("rl_feedback_log.txt")
