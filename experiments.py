"""Experiments: (1) separation quality vs SNR, ICA vs PCA, over several random mixing matrices;
(2) nonlinearity / algorithm comparison. Results -> results/snr_sweep.png, results/experiments.json"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audio_io import load_sources
from ica import ica, pca_components
from metrics import evaluate
from run import make_mixing_matrix

S, sr, names = load_sources("sources")
N = S.shape[0]
SNRS = [40, 30, 20, 10, 5, 0]
TRIALS = 10


def add_noise(X, snr_db, rng):
    p = np.mean(X ** 2, axis=1, keepdims=True)
    return X + rng.standard_normal(X.shape) * np.sqrt(p / 10 ** (snr_db / 10))


out = {"snr": SNRS, "ica": [], "pca": [], "ica_std": [], "pca_std": []}
for snr in SNRS:
    si, sp = [], []
    for trial in range(TRIALS):
        rng = np.random.default_rng(trial)
        X = add_noise(make_mixing_matrix(N, N, trial) @ S, snr, rng)
        si.append(evaluate(S, ica(X, n_components=N, seed=trial)[0])[0]["mean_si_sdr_db"])
        sp.append(evaluate(S, pca_components(X, N))[0]["mean_si_sdr_db"])
    out["ica"].append(float(np.mean(si))); out["ica_std"].append(float(np.std(si)))
    out["pca"].append(float(np.mean(sp))); out["pca_std"].append(float(np.std(sp)))
    print(f"SNR {snr:>2} dB | ICA {out['ica'][-1]:6.2f} +- {out['ica_std'][-1]:.2f} | "
          f"PCA {out['pca'][-1]:6.2f} +- {out['pca_std'][-1]:.2f}")

plt.figure(figsize=(6, 4))
plt.errorbar(SNRS, out["ica"], out["ica_std"], marker="o", label="ICA (FastICA)")
plt.errorbar(SNRS, out["pca"], out["pca_std"], marker="s", label="PCA only")
plt.xlabel("Input SNR (dB)"); plt.ylabel("Mean SI-SDR (dB)"); plt.grid(alpha=.3); plt.legend()
plt.title(f"Separation quality vs noise ({TRIALS} random mixing matrices)")
plt.tight_layout(); plt.savefig("results/snr_sweep.png", dpi=130)

out["variants"] = {}
for alg in ["symmetric", "deflation"]:
    for fun in ["tanh", "exp", "cube"]:
        r = []
        for trial in range(TRIALS):
            X = make_mixing_matrix(N, N, trial) @ S
            r.append(evaluate(S, ica(X, n_components=N, algorithm=alg, fun=fun, seed=trial)[0])[0]["mean_si_sdr_db"])
        out["variants"][f"{alg}/{fun}"] = float(np.mean(r))
        print(f"{alg:<10s} {fun:<5s} mean SI-SDR = {np.mean(r):6.2f} dB")
json.dump(out, open("results/experiments.json", "w"), indent=2)
