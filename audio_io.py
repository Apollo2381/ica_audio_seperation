import os, glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd


def load_wav(path, target_sr=None):
    sr, raw = wavfile.read(path)
    x = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        x /= np.iinfo(raw.dtype).max
    if x.ndim > 1:
        x = x.mean(axis=1)
    if target_sr and sr != target_sr:
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g)
        sr = target_sr
    return x, sr


def load_sources(folder, max_seconds=15):
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if not files:
        raise FileNotFoundError(f"No .wav files in {folder}")
    first_sr = load_wav(files[0])[1]
    sigs = [load_wav(f, first_sr)[0] for f in files]
    n = min(min(len(s) for s in sigs), int(max_seconds * first_sr))
    S = np.stack([s[:n] / (np.abs(s[:n]).max() + 1e-12) for s in sigs])
    return S, first_sr, [os.path.basename(f) for f in files]


def save_wav(path, x, sr):
    x = x / (np.abs(x).max() + 1e-12) * 0.9
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
