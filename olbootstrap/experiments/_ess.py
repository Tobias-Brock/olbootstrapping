import numpy as np


def ess_ewma(eta: float) -> float:
    """Exact effective sample size for EWMA smoothing.

    Args:
        eta (float): EWMA smoothing parameter (0 < eta <= 2).

    Returns:
        float: ESS = (2 - eta) / eta.
    """
    return (2.0 - float(eta)) / float(eta)


def ess_brown(eta: float) -> float:
    """Exact effective sample size for Brown's double exponential smoothing.

    Args:
        eta (float): Brown smoothing parameter.

    Returns:
        float: ESS = ((2 - eta)**3) / [eta * (5 eta^2 - 14 eta + 10)].
    """
    e = float(eta)
    num = (2.0 - e) ** 3
    den = e * (5.0 * e * e - 14.0 * e + 10.0)
    return num / den


def _holt_weights(eta: float, beta: float, K: int = 4000) -> np.ndarray:
    """Impulse-response weights for Holt's (double) exponential smoothing.

    Args:
        eta (float): Level smoothing parameter.
        beta (float): Trend smoothing parameter.
        K (int): Number of lags / length of weight vector (default 4000).

    Returns:
        np.ndarray: Weight vector of length K that sums to 1 (numerically).
    """
    w = np.zeros(K, dtype=float)
    for lag in range(K):
        s = 0.0
        tr = 0.0
        for t in range(K):
            inp = 1.0 if t == (K - 1 - lag) else 0.0
            prev_s = s
            s = eta * inp + (1.0 - eta) * (s + tr)
            tr = beta * (s - prev_s) + (1.0 - beta) * tr
        w[lag] = s
    w /= w.sum() if w.sum() != 0 else 1.0
    return w


def _holtwinters_weights(
    eta1: float, eta2: float, eta3: float, L: int, K: int = 8000
) -> np.ndarray:
    """Impulse-response weights for Holt-Winters forecast (level + season).

    Args:
        eta1 (float): Level smoothing parameter.
        eta2 (float): Trend smoothing parameter.
        eta3 (float): Seasonal smoothing parameter.
        L (int): Seasonal period.
        K (int): Number of lags / length of weight vector (default 8000).

    Returns:
        np.ndarray: Weight vector of length K (sums to 1 numerically).
    """
    w = np.zeros(K, dtype=float)
    for lag in range(K):
        s = 0.0
        tr = 0.0
        S = np.zeros(L, dtype=float)
        for t in range(K):
            inp = 1.0 if t == (K - 1 - lag) else 0.0
            idx = t % L
            S_prev = S[idx]
            Lt = eta1 * (inp - S_prev) + (1.0 - eta1) * (s + tr)
            Bt = eta2 * (Lt - s) + (1.0 - eta2) * tr
            St = eta3 * (inp - Lt) + (1.0 - eta3) * S_prev
            s, tr = Lt, Bt
            S[idx] = St
        idx_final = (K - 1) % L
        yhat = s + S[idx_final]
        w[lag] = yhat
    w /= w.sum() if w.sum() != 0 else 1.0
    return w


def effective_sample_size(
    smoothing_method: str,
    eta: float,
    beta: float | None = None,
    gamma: float | None = None,
    seasonal_period: int | None = None,
) -> float:
    """Estimate the effective sample size for a given smoothing method.

    Args:
        smoothing_method (str): Smoother name.
        eta (float): Primary smoothing parameter.
        beta (float | None): Trend smoothing parameter (used for Holt/HW).
        gamma (float | None): Seasonal smoothing parameter (used for HW).
        seasonal_period (int | None): Seasonal period for HW (defaults to 2).

    Returns:
        float: Estimated effective sample size (ESS).

    Raises:
        ValueError: If required parameters are invalid for the chosen method.
    """
    sm = (smoothing_method or '').lower()
    if sm == 'ewma':
        return ess_ewma(eta)
    if sm == 'brown':
        return ess_brown(eta)
    if sm == 'holt':
        b = float(beta) if beta is not None else max(1e-6, float(eta) / 4.0)
        w = _holt_weights(float(eta), b, K=4000)
        return 1.0 / float(np.sum(w**2))
    if sm in ('holtwinters', 'hw'):
        b = float(beta) if beta is not None else max(1e-6, float(eta) / 4.0)
        g = float(gamma) if gamma is not None else max(1e-6, float(eta) / 2.0)
        L = int(seasonal_period or 2)
        w = _holtwinters_weights(float(eta), b, g, L, K=max(4000, 4 * L))
        return 1.0 / float(np.sum(w**2))
    return ess_ewma(eta)
