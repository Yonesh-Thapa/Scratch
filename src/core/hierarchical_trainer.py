import numpy as np
from .probabilistic_encoder import ProbabilisticEncoder

class HierarchicalTrainer:
    """Stacked encoders/decoders for hierarchical feature learning."""
    def __init__(self, input_shape, sdr_dim, latent_dim, num_layers=2):
        self.input_shape = input_shape
        self.sdr_dim = sdr_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.encoders = [ProbabilisticEncoder(np.prod(input_shape) if i==0 else latent_dim, latent_dim) for i in range(num_layers)]
        self.decoders = [np.random.randn(latent_dim if i>0 else np.prod(input_shape), latent_dim) * 0.01 for i in range(num_layers)]

    def encode(self, x):
        zs = []
        mu_logvars = []
        for encoder in self.encoders:
            z, mu, logvar = encoder.encode(x)
            zs.append(z)
            mu_logvars.append((mu, logvar))
            x = z
        return zs, mu_logvars

    def decode(self, zs):
        x = zs[-1]
        for i in reversed(range(self.num_layers)):
            x = np.dot(self.decoders[i], x)
        return x.reshape(self.input_shape)

    @property
    def decoder(self):
        """Expose the main decoder weights for compatibility (input-level decoder)."""
        return self.decoders[0]

    @decoder.setter
    def decoder(self, value):
        self.decoders[0] = value
