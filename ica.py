"""ICA from scratch: centering, PCA whitening (covariance + eigendecomposition), FastICA.
Only NumPy is used. No library ICA anywhere."""
import numpy as np


def center(X):
    """Subtract the mean of each row (channel). X: (channels, samples)."""
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def whiten(Xc, n_components=None, eps=1e-12):
    """PCA whitening via covariance eigendecomposition.
    C = X X^T / n = E D E^T   ->   Z = D^{-1/2} E^T X   so that cov(Z) = I.
    Returns Z, whitening matrix V (k x channels), eigenvalues (descending)."""
    n = Xc.shape[1]
    C = (Xc @ Xc.T) / n
    d, E = np.linalg.eigh(C)             # symmetric matrix -> eigh
    order = np.argsort(d)[::-1]
    d, E = d[order], E[:, order]
    k = n_components or Xc.shape[0]
    d, E = np.maximum(d[:k], eps), E[:, :k]
    V = np.diag(d ** -0.5) @ E.T
    return V @ Xc, V, d


def _nonlinearity(u, fun):
    if fun == "tanh":                    # log-cosh contrast, robust default
        g = np.tanh(u)
        return g, 1.0 - g ** 2
    if fun == "exp":                     # Gaussian contrast, good for super-Gaussian
        e = np.exp(-u ** 2 / 2)
        return u * e, (1 - u ** 2) * e
    if fun == "cube":                    # kurtosis contrast
        return u ** 3, 3 * u ** 2
    raise ValueError(f"unknown nonlinearity {fun}")


def _sym_decorrelate(W):
    """W <- (W W^T)^{-1/2} W  (symmetric orthogonalisation)."""
    s, U = np.linalg.eigh(W @ W.T)
    return U @ np.diag(1.0 / np.sqrt(np.maximum(s, 1e-12))) @ U.T @ W


def fastica(Z, algorithm="symmetric", fun="tanh", max_iter=1000, tol=1e-7, seed=0):
    """FastICA fixed-point iteration on WHITENED data Z (k, n).
    Update: w <- E[z g(w^T z)] - E[g'(w^T z)] w, then re-orthogonalise.
    Returns S (k, n), W (k x k rotation on whitened data), iterations used."""
    k, n = Z.shape
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, k))
    iters = 0

    if algorithm == "symmetric":
        W = _sym_decorrelate(W)
        for iters in range(1, max_iter + 1):
            g, gp = _nonlinearity(W @ Z, fun)
            W_new = (g @ Z.T) / n - np.diag(gp.mean(axis=1)) @ W
            W_new = _sym_decorrelate(W_new)
            lim = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0))
            W = W_new
            if lim < tol:
                break
    elif algorithm == "deflation":
        for p in range(k):
            w = rng.standard_normal(k)
            w /= np.linalg.norm(w)
            for it in range(1, max_iter + 1):
                g, gp = _nonlinearity(w @ Z, fun)
                w_new = (Z * g).mean(axis=1) - gp.mean() * w
                w_new -= W[:p].T @ (W[:p] @ w_new)       # Gram-Schmidt
                w_new /= np.linalg.norm(w_new)
                lim = abs(abs(w_new @ w) - 1.0)
                w = w_new
                iters = max(iters, it)
                if lim < tol:
                    break
            W[p] = w
    else:
        raise ValueError("algorithm must be 'symmetric' or 'deflation'")
    return W @ Z, W, iters


def ica(X, n_components=None, **kw):
    """Full pipeline: center -> whiten -> FastICA. Returns S, total unmixing matrix, info."""
    Xc, mean = center(X)
    Z, V, eig = whiten(Xc, n_components)
    S, W, iters = fastica(Z, **kw)
    return S, W @ V, {"Z": Z, "eigenvalues": eig, "iterations": iters, "mean": mean}


def pca_components(X, n_components=None):
    """PCA baseline: just the whitened principal components (uncorrelated, not independent)."""
    Xc, _ = center(X)
    Z, _, _ = whiten(Xc, n_components)
    return Z


def kurtosis(S):
    """Excess kurtosis per row (0 for Gaussian)."""
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    return (S ** 4).mean(axis=1) - 3.0
