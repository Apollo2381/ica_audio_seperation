"""Evaluation that handles ICA's permutation, sign and scale ambiguities."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match(S_true, S_est):
    """Match estimates to originals with the Hungarian algorithm on |correlation|.
    Returns permutation idx (est row for each true row) and the |corr| matrix."""
    n = S_true.shape[0]
    C = np.abs(np.corrcoef(S_true, S_est)[:n, n:])
    rows, cols = linear_sum_assignment(-C)
    return cols[np.argsort(rows)], C


def si_sdr(s, s_hat):
    """Scale-invariant SDR (dB). The optimal gain alpha (sign included) is fitted first:
    alpha = <s_hat, s> / <s_hat, s_hat>;  SDR = 10 log10(||s||^2 / ||s - alpha s_hat||^2)."""
    s, s_hat = s - s.mean(), s_hat - s_hat.mean()
    alpha = (s_hat @ s) / (s_hat @ s_hat + 1e-20)
    err = s - alpha * s_hat
    return 10 * np.log10((s @ s) / (err @ err + 1e-20))


def evaluate(S_true, S_est):
    perm, C = match(S_true, S_est)
    est = S_est[perm]
    corr = [float(np.corrcoef(S_true[i], est[i])[0, 1]) for i in range(len(perm))]
    sdr = [float(si_sdr(S_true[i], est[i])) for i in range(len(perm))]
    return {"perm": perm.tolist(), "abs_corr": [abs(c) for c in corr], "signed_corr": corr,
            "si_sdr_db": sdr, "mean_abs_corr": float(np.mean(np.abs(corr))),
            "mean_si_sdr_db": float(np.mean(sdr))}, est
