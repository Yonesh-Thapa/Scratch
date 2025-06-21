import numpy as np

class ProbabilisticEncoder:
    """Variational-style probabilistic encoder for images."""
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Simple linear weights for mean and logvar
        self.W_mu = np.random.randn(latent_dim, input_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(latent_dim, input_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

    def encode(self, x):
        x = x.flatten()
        mu = np.dot(self.W_mu, x) + self.b_mu
        logvar = np.dot(self.W_logvar, x) + self.b_logvar
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z, mu, logvar

    def kl_divergence(self, mu, logvar):
        return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
