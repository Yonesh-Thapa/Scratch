"""
brain_visualizer.py
Visualizes neuron activations and input/reconstruction images live using matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt

class BrainVisualizer:
    """Visualizes input, reconstruction, neuron activations, attention, and temporal/context vectors for each layer."""
    def __init__(self, input_shape, sdr_dim, context_dim=64):
        self.input_shape = input_shape
        self.sdr_dim = sdr_dim
        self.context_dim = context_dim
        self.fig, self.axs = plt.subplots(1, 6, figsize=(18, 3))
        self.im_input = self.axs[0].imshow(np.zeros(input_shape), cmap='gray', vmin=0, vmax=1)
        self.axs[0].set_title('Input')
        self.im_recon = self.axs[1].imshow(np.zeros(input_shape), cmap='gray', vmin=0, vmax=1)
        self.axs[1].set_title('Reconstruction')
        self.im_error = self.axs[2].imshow(np.zeros(input_shape), cmap='bwr', vmin=-1, vmax=1)
        self.axs[2].set_title('Error')
        self.im_sdr = self.axs[3].imshow(np.zeros((1, sdr_dim)), aspect='auto', cmap='Greens', vmin=0, vmax=1)
        self.axs[3].set_title('SDR')
        self.im_attn = self.axs[4].imshow(np.zeros(input_shape), cmap='hot', vmin=0, vmax=1)
        self.axs[4].set_title('Attention')
        self.im_context = self.axs[5].imshow(np.zeros((1, context_dim)), aspect='auto', cmap='Blues', vmin=-1, vmax=1)
        self.axs[5].set_title('Context')
        self.letter_text = self.fig.text(0.5, 0.95, '', ha='center', va='top', fontsize=16, color='blue')
        for ax in self.axs:
            ax.axis('off')
        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, input_img, recon_img, error_img, sdr, attn_map, context_vec, label=None):
        self.im_input.set_data(input_img)
        self.im_recon.set_data(recon_img)
        self.im_error.set_data(error_img)
        self.im_sdr.set_data(sdr.reshape(1, -1))
        self.im_attn.set_data(attn_map)
        self.im_context.set_data(context_vec.reshape(1, -1))
        if label is not None:
            self.letter_text.set_text(f'Letter: {label}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
