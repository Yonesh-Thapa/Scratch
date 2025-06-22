"""
rl_log_visualizer.py
Visualize RL log: error, reward, plasticity, valence, penalty counts, diversity, and forced exploration events.
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
from module_status import print_module_status

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

def plot_rl_log(log_path, baseline_path=None, save_path=None, show=True, highlight_drops=True):
    df = parse_log(log_path)
    letters = sorted(df['letter'].unique())
    epochs = sorted(df['epoch'].unique())
    fig, axs = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    for letter in letters[:5]:
        ldf = df[df['letter'] == letter]
        axs[0].plot(ldf['epoch'], ldf['error'], label=letter)
        axs[1].plot(ldf['epoch'], ldf['reward'], label=letter)
        axs[2].plot(ldf['epoch'], ldf['valence_ma'], label=letter)
        axs[3].plot(ldf['epoch'], ldf['diversity'], label=letter)
        axs[4].plot(ldf['epoch'], ldf['lr_mod'], label=letter)
        # Highlight drops/improvements
        if highlight_drops:
            drops = ldf[ldf['error'].diff() > 0.05]
            axs[0].scatter(drops['epoch'], drops['error'], color='red', s=40, marker='v', label=f'{letter} drop')
            improvements = ldf[ldf['error'].diff() < -0.05]
            axs[0].scatter(improvements['epoch'], improvements['error'], color='green', s=40, marker='^', label=f'{letter} improve')
    axs[0].set_ylabel("Error")
    axs[1].set_ylabel("Reward")
    axs[2].set_ylabel("ValenceMA")
    axs[3].set_ylabel("Diversity")
    axs[4].set_ylabel("Learning Rate")
    axs[4].set_xlabel("Epoch")
    for ax in axs:
        ax.legend()
    if baseline_path:
        bdf = parse_log(baseline_path)
        for letter in letters[:5]:
            ldf = bdf[bdf['letter'] == letter]
            axs[0].plot(ldf['epoch'], ldf['error'], '--', label=f'{letter} (baseline)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

if __name__ == "__main__":
    print_module_status()
    import argparse
    parser = argparse.ArgumentParser(description="RL Log Visualizer: Plot error, reward, plasticity, valence, diversity, and more.")
    parser.add_argument('log', type=str, nargs='?', default='rl_feedback_log.txt', help='Path to RL log file')
    parser.add_argument('--baseline', type=str, default=None, help='Optional baseline log for comparison')
    parser.add_argument('--save', type=str, default=None, help='Path to save plot image (PNG)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot interactively')
    parser.add_argument('--no-highlight', action='store_true', help='Do not highlight drops/improvements')
    args = parser.parse_args()
    plot_rl_log(args.log, baseline_path=args.baseline, save_path=args.save, show=not args.no_show, highlight_drops=not args.no_highlight)
# USAGE:
# python rl_log_visualizer.py rl_feedback_log.txt --baseline baseline_log.txt --save out.png
