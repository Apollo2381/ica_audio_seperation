# Unmix the Signal: ICA from Scratch

Blind source separation of mixed audio using **Independent Component Analysis implemented from scratch**
(NumPy only; no library ICA). Pipeline: mix → center → PCA whitening (covariance + eigendecomposition)
→ FastICA → export WAVs → evaluate (correlation, SI-SDR) with permutation/sign/scale handled.

## Audio sources
<!-- REPLACE with your real clips and link each one -->
| File | Type | Source / licence |
|------|------|------------------|
| `sources/speech.wav` | speech | _link here_ |
| `sources/music.wav` | music | _link here_ |
| `sources/ambient.wav` | ambient | _link here_ |

> The repo ships with synthetic stand-ins from `make_demo_sources.py`. Replace them with real clean clips
> (any sample rate; they are resampled, mono-mixed and trimmed automatically) before submitting.
> At most one source may be Gaussian-like; rain or white noise as *several* sources will not separate.

## Setup and run
```bash
pip install -r requirements.txt
python run.py                      # mixes, separates, exports WAVs, prints metrics, saves plots
python run.py --algorithm deflation --fun exp --n_mix 4   # variants (more mixtures than sources is fine)
python experiments.py              # SNR sweep (ICA vs PCA) + nonlinearity comparison
jupyter notebook notebook.ipynb    # walkthrough with plots and audio players
```

## Layout
| Path | Purpose |
|------|---------|
| `ica.py` | `center`, `whiten` (cov + `eigh`), `fastica` (symmetric and deflation; tanh/exp/cube), kurtosis |
| `metrics.py` | Hungarian matching, correlation, scale-invariant SDR |
| `audio_io.py` | WAV loading/resampling/saving |
| `run.py` | Main pipeline |
| `experiments.py` | SNR sweep, PCA baseline, nonlinearity comparison |
| `results/` | `mixtures/`, `separated/`, `separated_pca/` WAVs, plots, `metrics.json` |

## Method
1. **Mix:** X = A·S with a random well-conditioned matrix A (`cond(A) < 20`).
2. **Center:** subtract each channel's mean.
3. **Whiten:** C = XXᵀ/n = EDEᵀ, Z = D^(-1/2)EᵀX, so cov(Z) = I.
4. **FastICA:** w ← E[z·g(wᵀz)] − E[g′(wᵀz)]·w, then re-orthogonalise (symmetric: (WWᵀ)^(-1/2)W; deflation: Gram-Schmidt).
5. **Recover** S = W·Z and export as 16-bit WAV (sign-corrected, peak-normalised).
6. **Evaluate:** match estimates to originals by the Hungarian algorithm on |corr| (permutation), fit the optimal
   scalar gain including sign (scale/sign), then report |corr| and SI-SDR.

## Results
<!-- PASTE your numbers after running with your REAL audio -->
Run `python run.py` and `python experiments.py`, then paste the tables here. Figures:
![waveforms](results/waveforms.png)
![stages](results/scatter_stages.png)
![snr](results/snr_sweep.png)

## Be ready to explain
1. **ICA vs PCA.** PCA gives *uncorrelated* components using second-order statistics only. Any rotation of whitened data is still uncorrelated, so PCA can't pick the right one. ICA uses higher-order statistics to find the rotation giving *independent* components.
2. **Whitening.** Makes the data uncorrelated with unit variance, so the unmixing matrix reduces to an orthogonal rotation (n(n−1)/2 free parameters instead of n²) and the FastICA update becomes simple and well-conditioned.
3. **Non-Gaussianity and the CLT.** A sum of independent variables is more Gaussian than its parts, so mixtures are more Gaussian than sources. Maximising non-Gaussianity (kurtosis/negentropy proxies) recovers sources. Gaussian sources are rotation-invariant, so at most one can be separated.
4. **Experiments and metrics.** Correlation and SI-SDR are invariant to scale and (after matching) permutation and sign, which ICA cannot determine. The SNR sweep and the PCA baseline show how much of the gain comes from higher-order statistics and how noise erodes it; nonlinearity/algorithm variants test robustness of the fixed-point update.

## Limitations
Instantaneous (not convolutive) mixing only; real multi-device recordings have delays and reverb, which break the linear model. Noise is not modelled (noiseless ICA), so SDR falls with SNR.
