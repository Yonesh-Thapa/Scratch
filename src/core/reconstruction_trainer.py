"""
reconstruction_trainer.py
Manages the see→draw→compare→reinforce cycle.
Upgraded: proportional plasticity, min/max lr, docstrings.
"""
import numpy as np
from .feature_extractor import FeatureExtractor
from .factorial_encoder import FactorialEncoder
from .letter_decoder import LetterDecoder
from .hierarchical_trainer import HierarchicalTrainer
from .attention import Attention
from .temporal_context import TemporalContext

class ReconstructionTrainer:
    """Manages the see→draw→compare→reinforce cycle for training, with hierarchy, attention, temporal context, and adaptive plasticity."""
    def __init__(self, input_shape, sdr_dim, sparsity=0.05, lr=0.01, min_lr=1e-5, max_lr=0.1, alpha=0.1, latent_dim=64, num_layers=2, num_heads=1, hidden_dim=64):
        """
        Args:
            input_shape: shape of input images
            sdr_dim: number of SDR neurons
            sparsity: SDR sparsity
            lr: initial learning rate
            min_lr: minimum learning rate
            max_lr: maximum learning rate
            alpha: plasticity modulation factor
            latent_dim: dimensionality of latent space for hierarchical module
            num_layers: number of layers in the hierarchical module
            num_heads: number of attention heads
            hidden_dim: dimensionality of hidden layers in temporal context module
        """
        self.input_shape = input_shape
        self.sdr_dim = sdr_dim
        self.sparsity = sparsity
        self.base_lr = lr
        self.lr = {letter: lr for letter in getattr(self, 'letters', [])}  # Will be set externally
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.alpha = alpha
        self.feature_extractor = FeatureExtractor()
        self.attention = Attention(input_shape, num_heads=4, init_scale=2.0, temperature=2.0)
        self.hierarchical = HierarchicalTrainer(input_shape, sdr_dim, latent_dim, num_layers)
        self.temporal = TemporalContext(latent_dim, hidden_dim)
        self.sdr_activity = np.zeros(sdr_dim)  # Track neuron usage

    def train_step(self, image: np.ndarray, letter=None):
        """
        Performs one see→draw→compare→reinforce cycle.
        Args:
            image (np.ndarray): 2D input image.
        Returns:
            dict: {input, features, sdr, reconstruction, error}
        """
        features = self.feature_extractor.extract(image)
        flat_features = features.flatten()
        # Apply attention to the original image, not the flattened features
        attended, attn_map = self.attention.apply_attention(image)
        # Robust fallback: ensure attended is the correct shape
        expected_shape = self.input_shape
        if attended.shape == (1,):
            # Broadcast scalar/1D to image shape
            attended = np.full(expected_shape, attended.item())
        elif attended.ndim == 1:
            # If attended is 1D (e.g., (num_heads,)), broadcast or average to image shape
            if attended.shape[0] == np.prod(expected_shape):
                attended = attended.reshape(expected_shape)
            else:
                # Broadcast each head's value across a portion of the image, or average if ambiguous
                # Here, we simply fill the image with the mean value as a robust fallback
                attended = np.full(expected_shape, attended.mean())
        elif attended.shape != expected_shape:
            raise ValueError(f"Attention output shape {attended.shape} does not match expected {expected_shape}")
        # Diagnostic: print shapes for debugging
        print(f"attended shape: {attended.shape}")
        flat_attended = attended.flatten()
        print(f"flat_attended shape: {flat_attended.shape}")
        # Check if the flattened attended image matches the expected input size
        expected_size = np.prod(self.input_shape)
        if flat_attended.shape[0] != expected_size:
            raise ValueError(f"Expected flattened attended shape ({expected_size},) but got {flat_attended.shape}")
        zs, mu_logvars = self.hierarchical.encode(flat_attended)
        context = self.temporal.step(zs[-1])
        # --- Learning step: update decoder weights using error ---
        # Forward pass
        reconstruction = self.hierarchical.decode(zs).reshape(self.input_shape)
        error = image - reconstruction
        # Use per-letter learning rate if available
        lr = self.lr[letter] if letter is not None and letter in self.lr else self.base_lr
        # Simple gradient update for the first decoder (input-level)
        # dL/dW = -2 * error * z^T (outer product)
        z_last = zs[-1]
        grad = -2 * np.outer(error.flatten(), z_last)
        self.hierarchical.decoders[0] -= lr * grad
        # --- Attention learning: update attention weights ---
        # Compute gradient of loss w.r.t. attn_map (approximate: error * input)
        if attn_map is not None:
            grad_attn_map = -2 * (image - reconstruction) * image  # shape: (H, W)
            if attn_map.ndim == 3:
                grad_attn_map = np.broadcast_to(grad_attn_map, attn_map.shape)
            else:
                grad_attn_map = grad_attn_map[None, ...]  # add head dim
            # Ensure grad_attn_map matches the number of heads for update
            if grad_attn_map.shape[0] != self.attention.num_heads:
                grad_attn_map = np.broadcast_to(grad_attn_map, (self.attention.num_heads, *grad_attn_map.shape[1:]))
            self.attention.update(image, grad_attn_map)
        # Optionally update hierarchical/attention/temporal modules here
        return {
            'input': image,
            'features': features,
            'attended': attended,
            'attn_map': attn_map,
            'zs': zs,
            'context': context,
            'reconstruction': reconstruction,
            'error': error
        }

    def set_learning_rate(self, letter, lr):
        """Set the learning rate for a specific letter."""
        self.lr[letter] = max(self.min_lr, min(self.max_lr, lr))

    def get_learning_rate(self, letter):
        """Get the current learning rate for a specific letter."""
        return self.lr.get(letter, self.base_lr)

    def get_base_lr(self, letter=None):
        """Get the base learning rate (before modulation)."""
        return self.base_lr

    def grow_neurons(self, n_new):
        self.encoder.grow(n_new)
        self.hierarchical.decoder.grow(n_new)
        self.sdr_activity = np.concatenate([self.sdr_activity, np.zeros(n_new)])
        self.sdr_dim += n_new

    def prune_neurons(self, min_activity=1):
        inactive = np.where(self.sdr_activity < min_activity)[0]
        if len(inactive) > 0:
            self.encoder.prune(inactive)
            self.hierarchical.decoder.prune(inactive)
            self.sdr_activity = np.delete(self.sdr_activity, inactive)
            self.sdr_dim = self.encoder.sdr_dim

    def rewire_neurons(self, indices):
        self.encoder.rewire(indices)

    def set_plasticity(self, reward=0.0):
        """
        Modulate the learning rate (plasticity) by a factor proportional to reward.
        lr *= (1 + alpha * reward), clamped to [min_lr, max_lr].
        """
        self.lr *= (1 + self.alpha * reward)
        self.lr = max(min(self.lr, self.max_lr), self.min_lr)

    def visualize_step(self, visualizer, result, label=None):
        """Convenience method to visualize a training step with all advanced outputs."""
        # Ensure attn_map is 2D for visualization
        attn_map = result['attn_map']
        if attn_map is not None and attn_map.ndim == 3 and attn_map.shape[0] == 1:
            attn_map = attn_map.squeeze(0)
        # Debug: print attention map stats
        if attn_map is not None:
            print(f"[Attention Debug] attn_map min: {attn_map.min():.6f}, max: {attn_map.max():.6f}, mean: {attn_map.mean():.6f}, shape: {attn_map.shape}")
        visualizer.update(
            input_img=result['input'],
            recon_img=result['reconstruction'],
            error_img=result['error'],
            sdr=result['zs'][-1],
            attn_map=attn_map,
            context_vec=result['context'],
            label=label
        )
