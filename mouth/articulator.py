import numpy as np

class DigitalArticulator:
    """
    Simulates a digital mouth that generates waveforms from control parameters.
    """
    def __init__(self, wave_dim=8000, sr=16000, noise_std=0.01):
        self.wave_dim = wave_dim
        self.sr = sr
        self.noise_std = noise_std

    def synthesize(self, controls):
        t = np.linspace(0, self.wave_dim / self.sr, self.wave_dim)
        pitch = controls.get('pitch', 220)
        amp = controls.get('amplitude', 0.5)
        f1 = controls.get('formant1', 700)
        f2 = controls.get('formant2', 1200)
        wave = amp * (np.sin(2 * np.pi * pitch * t) + 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t))
        wave += np.random.normal(0, self.noise_std, wave.shape)
        return wave.astype(np.float32)
