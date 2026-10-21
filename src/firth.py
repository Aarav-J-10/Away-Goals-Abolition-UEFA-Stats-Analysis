"""Firth (1993) penalised-likelihood logistic regression with profile penalised-likelihood CIs and
penalised likelihood-ratio tests (the approach of R's logistf; Heinze & Schemper 2002).
Used for the rare outcome (penalty shoot-outs)."""
import numpy as np
from scipy import stats, optimize


def _pll(X, y, b):
    eta = np.clip(X @ b, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    _, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * logdet


def _fit(X, y, fixed=None, b0=None, maxit=300, tol=1e-10):
    """Maximise the penalised log-likelihood; `fixed` = (index, value) holds one coefficient constant."""
    k = X.shape[1]
    b = np.zeros(k) if b0 is None else b0.copy()
    free = np.arange(k)
    if fixed is not None:
        b[fixed[0]] = fixed[1]
        free = np.array([i for i in range(k) if i != fixed[0]])
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        I = X.T @ (X * W[:, None])
        inv = np.linalg.inv(I)
        sw = np.sqrt(W)[:, None] * X
        h = np.einsum("ij,jk,ik->i", sw, inv, sw)
        U = X.T @ (y - p + h * (0.5 - p))
        step = np.zeros(k)
        step[free] = np.linalg.solve(I[np.ix_(free, free)], U[free])
        ll0, t = _pll(X, y, b), 1.0
        while t > 1e-8:
            nb = b + t * step
            if _pll(X, y, nb) >= ll0 - 1e-10:
                break
            t /= 2
        b = nb
        if np.max(np.abs(t * step)) < tol:
            break
    return b


def firth_logit(X, y, names):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = _fit(X, y)
    ll_full = _pll(X, y, b)
    out = []
    for j, nm in enumerate(names):
        prof = lambda bj: _pll(X, y, _fit(X, y, fixed=(j, bj), b0=b))
        lr = 2 * (ll_full - prof(0.0))
        pval = stats.chi2.sf(max(lr, 0), 1)
        crit = ll_full - stats.chi2.ppf(0.95, 1) / 2
        f = lambda bj: prof(bj) - crit
        lo = optimize.brentq(f, b[j] - 12, b[j], xtol=1e-6) if f(b[j] - 12) < 0 else -np.inf
        hi = optimize.brentq(f, b[j], b[j] + 12, xtol=1e-6) if f(b[j] + 12) < 0 else np.inf
        out.append(dict(term=nm, coef=b[j], OR=np.exp(b[j]), ci_low=np.exp(lo), ci_high=np.exp(hi), p=pval, lr_chi2=lr))
    return out
