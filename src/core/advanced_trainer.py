import numpy as np
from src.core.hierarchical_trainer import HierarchicalTrainer
from src.core.attention import Attention
from src.core.temporal_context import TemporalContext

class AdvancedTrainer:
    """Full modular trainer with hierarchy, attention, temporal context, and probabilistic reasoning."""
    def __init__(self, input_shape, sdr_dim, latent_dim, num_layers=2, num_heads=1, hidden_dim=64):
        self.input_shape = input_shape
        self.sdr_dim = sdr_dim
        self.latent_dim = latent_dim
        self.hierarchical = HierarchicalTrainer(input_shape, sdr_dim, latent_dim, num_layers)
        self.attention = Attention(input_shape, num_heads)
        self.temporal = TemporalContext(latent_dim, hidden_dim)

    def train_step(self, x):
        # Apply attention
        attended, attn_map = self.attention.apply_attention(x)
        # Hierarchical encode
        zs, mu_logvars = self.hierarchical.encode(attended)
        # Temporal context
        context = self.temporal.step(zs[-1])
        # Hierarchical decode
        recon = self.hierarchical.decode(zs)
        # Compute error
        error = x - recon
        return {
            'input': x,
            'attended': attended,
            'attn_map': attn_map,
            'zs': zs,
            'context': context,
            'reconstruction': recon,
            'error': error
        }
