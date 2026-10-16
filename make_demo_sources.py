"""Synthesises 3 stand-in sources (speech-like, music-like, ambient-like) so the pipeline runs
end-to-end. REPLACE these with real clean recordings in sources/ for the actual submission."""
import numpy as np, os
from scipy.io import wavfile
from scipy.signal import lfilter

sr, dur = 16000, 12
t = np.arange(sr * dur) / sr
rng = np.random.default_rng(1)

# speech-like: pitch-glided harmonic stack, vowel-ish formant weighting, syllabic gating
f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
phase = 2 * np.pi * np.cumsum(f0) / sr
speech = sum(np.sin(h * phase) * np.exp(-((h * f0 - 700) / 500) ** 2) for h in range(1, 30))
gate = np.clip(np.sin(2 * np.pi * 3.1 * t) + 0.3 * np.sin(2 * np.pi * 0.9 * t), 0, None) ** 1.5
speech = speech * gate + 0.01 * rng.standard_normal(len(t))

# music-like: pentatonic melody, decaying plucked notes
notes = [262, 294, 330, 392, 440, 392, 330, 294]
music = np.zeros(len(t))
step = int(sr * 0.35)
for i in range(0, len(t) - step * 2, step):
    f = notes[(i // step) % len(notes)]
    tt = np.arange(step * 2) / sr
    env = np.exp(-4 * tt)
    music[i:i + step * 2] += env * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt))

# ambient-like: rain-ish sparse crackle through a low-pass (impulsive => super-Gaussian)
clicks = (rng.random(len(t)) < 0.004) * rng.standard_normal(len(t)) * 4
ambient = lfilter([1], [1, -0.85], clicks) + 0.02 * rng.standard_normal(len(t))

os.makedirs("sources", exist_ok=True)
for name, x in [("speech", speech), ("music", music), ("ambient", ambient)]:
    x = x / np.abs(x).max() * 0.9
    wavfile.write(f"sources/{name}.wav", sr, (x * 32767).astype(np.int16))
print("Wrote demo sources to sources/")
