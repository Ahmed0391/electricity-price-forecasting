"""
Schwartz factor models for electricity spot prices.

Implements the one-factor Ornstein-Uhlenbeck model (OLS and MLE
calibration) and the two-factor model estimated with a Kalman filter.

Theory
------
The log-price ``X_t = ln(S_t + shift)`` follows an Ornstein-Uhlenbeck
process ``dX_t = a(m - X_t) dt + sigma dW_t``, whose exact discretization
is an AR(1): ``X_i = phi0 + phi1 X_{i-1} + eps_i`` with ``phi1 = exp(-a h)``.
The two-factor model splits the log-price into a mean-reverting short-term
component ``c_t`` and a random-walk long-term component ``l_t``.
"""

import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# One-factor model
# ---------------------------------------------------------------------------

def calibrate_ols(X, h):
    """Calibrate the one-factor model by ordinary least squares.

    The AR(1) coefficients are estimated by OLS, then mapped back to the
    physical parameters ``(a, m, sigma)``.

    Parameters
    ----------
    X : numpy.ndarray
        Log-price series.
    h : float
        Time step (1 for daily data, 1/365 for annualized).

    Returns
    -------
    dict
        Keys: ``a``, ``m``, ``sigma``, ``phi0``, ``phi1``, ``residuals``.
    """
    n = len(X)
    A = np.column_stack([np.ones(n - 1), X[:-1]])
    B = X[1:]

    phi0, phi1 = inv(A.T @ A) @ A.T @ B
    residuals = B - A @ np.array([phi0, phi1])
    sigma_eps = np.std(residuals)

    a = -np.log(phi1) / h
    m = phi0 / (1 - phi1)
    sigma = sigma_eps * np.sqrt(2 * a / (1 - np.exp(-2 * a * h)))

    return {"a": a, "m": m, "sigma": sigma,
            "phi0": phi0, "phi1": phi1, "residuals": residuals}


def _neg_log_likelihood_1f(theta, X):
    """Negative log-likelihood of the Gaussian AR(1) model."""
    phi0, phi1, phi2 = theta
    n = len(X) - 1
    resid = X[1:] - phi1 * X[:-1] - phi0
    ll = -n / 2 * np.log(2 * np.pi) - n / 2 * phi2 \
        - 0.5 * np.sum(resid ** 2) * np.exp(-phi2)
    return -ll


def calibrate_mle(X, h):
    """Calibrate the one-factor model by maximum likelihood.

    For the Gaussian AR(1), MLE is equivalent to OLS; this routine mainly
    serves to verify that equivalence numerically. The optimizer is
    initialized from naive moment estimators.

    Returns
    -------
    dict
        Same structure as :func:`calibrate_ols`.
    """
    x_bar = np.mean(X)
    phi1_init = 0.5 * ((X[2] - x_bar) / (X[1] - x_bar) - (X[1] - x_bar) / x_bar)
    phi0_init = x_bar * (1 - phi1_init)
    resid_init = X[1:] - phi1_init * X[:-1] - phi0_init
    phi2_init = np.log(np.var(resid_init))

    res = minimize(_neg_log_likelihood_1f, [phi0_init, phi1_init, phi2_init],
                   args=(X,), method="Nelder-Mead")
    phi0, phi1, phi2 = res.x

    a = -np.log(phi1) / h
    m = phi0 / (1 - phi1)
    sigma_eps = np.sqrt(np.exp(phi2))
    sigma = sigma_eps * np.sqrt(2 * a / (1 - np.exp(-2 * a * h)))
    residuals = X[1:] - phi1 * X[:-1] - phi0

    return {"a": a, "m": m, "sigma": sigma,
            "phi0": phi0, "phi1": phi1, "residuals": residuals}


def predict_1f(params, X0, horizon, h):
    """Multi-step forecast with 95% prediction interval.

    Uses the closed-form conditional mean and variance of the OU process.
    The forecast converges towards ``m`` at rate ``a``.

    Parameters
    ----------
    params : dict
        Calibrated parameters (must contain ``a``, ``m``, ``sigma``).
    X0 : float
        Last observed log-price (forecast origin).
    horizon : int
        Number of steps to forecast.
    h : float
        Time step.

    Returns
    -------
    dict
        Keys ``mean``, ``lower``, ``upper`` (arrays of length ``horizon``).
    """
    a, m, sigma = params["a"], params["m"], params["sigma"]
    taus = np.arange(1, horizon + 1) * h

    mean = m + (X0 - m) * np.exp(-a * taus)
    var = sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * taus))
    std = np.sqrt(var)

    return {"mean": mean, "lower": mean - 1.96 * std, "upper": mean + 1.96 * std}


# ---------------------------------------------------------------------------
# Two-factor model + Kalman filter
# ---------------------------------------------------------------------------

def get_discrete_params(a, m, sigma_c, sigma_l, h):
    """Exact discretization coefficients of the two-factor model.

    Returns
    -------
    tuple
        ``(phi1, phi0_c, phi0_l, Qc, Ql)`` where ``phi1 = exp(-a h)`` is the
        short-term AR coefficient, the ``phi0`` are drift constants, and
        ``Qc``/``Ql`` are the exact state-noise variances.
    """
    phi1 = np.exp(-a * h)
    phi0_c = m * (1 - phi1)
    phi0_l = -0.5 * sigma_l ** 2 * h
    Qc = sigma_c ** 2 * (1 - np.exp(-2 * a * h)) / (2 * a)
    Ql = sigma_l ** 2 * h
    return phi1, phi0_c, phi0_l, Qc, Ql


def kalman_filter(X, params, h=1 / 365, R=1e-4):
    """Run the Kalman filter for the two-factor model.

    State ``alpha = [c, l]`` evolves as ``alpha_i = F alpha_{i-1} + d + eta``,
    and only the sum ``X_i = c_i + l_i`` is observed (``H = [1, 1]``). The
    observation noise ``R`` is near zero because spot prices carry no
    measurement error.

    Parameters
    ----------
    X : numpy.ndarray
        Observed log-price series.
    params : sequence
        ``(a, m, sigma_c, sigma_l)``.
    h : float
        Time step.
    R : float
        Observation noise variance (kept small but positive for stability).

    Returns
    -------
    tuple
        ``(alpha, innovations, S)`` where ``alpha`` has shape ``(n, 2)``,
        ``innovations`` and ``S`` have length ``n - 1``.
    """
    a, m, sigma_c, sigma_l = params
    phi1, phi0_c, phi0_l, Qc, Ql = get_discrete_params(a, m, sigma_c, sigma_l, h)

    F = np.array([[phi1, 0.0], [0.0, 1.0]])
    d = np.array([phi0_c, phi0_l])
    H = np.array([[1.0, 1.0]])
    Q = np.diag([Qc, Ql])

    n = len(X)
    alpha = np.zeros((n, 2))
    innovations = np.zeros(n - 1)
    S_list = np.zeros(n - 1)

    # Constrained initialization: all of X[0] attributed to c, high prior on l
    alpha[0] = [X[0], 0.0]
    P = np.diag([Qc, 100.0])

    for i in range(1, n):
        # Prediction
        a_pred = F @ alpha[i - 1] + d
        P_pred = F @ P @ F.T + Q
        # Innovation
        v = X[i] - (H @ a_pred)[0]
        S = (H @ P_pred @ H.T)[0, 0] + R
        # Kalman gain
        K = (P_pred @ H.T) / S
        # Update
        alpha[i] = a_pred + K.flatten() * v
        P = (np.eye(2) - K @ H) @ P_pred
        innovations[i - 1] = v
        S_list[i - 1] = S

    return alpha, innovations, S_list


def _neg_log_likelihood_2f(params, X, h):
    """Negative log-likelihood of the Kalman innovations."""
    a, m, sigma_c, sigma_l = params
    if a <= 0 or sigma_c <= 0 or sigma_l <= 0:
        return 1e10
    try:
        _, innov, S = kalman_filter(X, params, h)
        ll = -0.5 * np.sum(np.log(np.abs(S)) + innov ** 2 / S)
        return -ll
    except Exception:
        return 1e10


def calibrate_2f(X, h=1 / 365, init=None):
    """Calibrate the two-factor model by maximizing the innovation likelihood.

    The likelihood has no closed form (each evaluation runs the full Kalman
    filter), so a derivative-free Nelder-Mead optimizer is used.

    Parameters
    ----------
    X : numpy.ndarray
        Log-price series.
    h : float
        Time step.
    init : sequence, optional
        Initial ``(a, m, sigma_c, sigma_l)``. Defaults to a naive guess;
        initializing from the one-factor fit improves convergence.

    Returns
    -------
    dict
        Calibrated parameters plus filtered components ``c_hat``, ``l_hat``,
        the innovations, and the optimizer success flag.
    """
    if init is None:
        init = [50.0, np.mean(X), 0.5, 0.2]

    result = minimize(_neg_log_likelihood_2f, init, args=(X, h),
                      method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})

    a, m, sc, sl = result.x
    alpha, innov, S = kalman_filter(X, result.x, h)

    return {"a": a, "m": m, "sigma_c": sc, "sigma_l": sl,
            "c_hat": alpha[:, 0], "l_hat": alpha[:, 1],
            "innov": innov, "S": S, "success": result.success}


def half_life(a):
    """Mean-reversion half-life ``ln(2) / a`` (same time unit as ``a``)."""
    return np.log(2) / a
