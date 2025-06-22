import numpy as np
import scipy.fftpack

class DigitalCochlea:
    """
    Simulates a digital cochlea (filterbank) for audio processing.
    """
    def __init__(self, n_filters=13, sr=16000, noise_std=0.01):
        self.n_filters = n_filters
        self.sr = sr
        self.noise_std = noise_std

    def process(self, waveform):
        frame_size = self.sr // 100  # 10ms frames
        n_frames = len(waveform) // frame_size
        features = []
        for i in range(n_frames):
            frame = waveform[i*frame_size:(i+1)*frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size-len(frame)))
            dct = scipy.fftpack.dct(frame, norm='ortho')[:self.n_filters]
            features.append(dct)
        if not features:
            features = np.zeros((1, self.n_filters))
        else:
            features = np.stack(features)
        features += np.random.normal(0, self.noise_std, features.shape)
        return np.mean(features, axis=0)
