"""
Hybrid Kalman + Machine Learning forecasting.

This module implements the core contribution of the project: instead of
forecasting the raw log-price, the Kalman filter first decomposes it into a
short-term mean-reverting component ``c_t`` and a long-term random-walk
component ``l_t`` (see :mod:`schwartz`). A dedicated ML model is then trained
on each component, and their predictions are recombined as
``X_hat = c_hat + l_hat``.

The intuition is that each component carries a simpler, more homogeneous
signal than the raw price, so a specialized model learns it more easily.
Empirically this improves accuracy for both XGBoost and LSTM back-ends.
"""

import numpy as np
import pandas as pd

from schwartz import kalman_filter
from ml_models import (
    build_features_short, build_features_long,
    train_xgboost, train_lstm, lstm_predict,
)


def decompose(X, params, h=1 / 365):
    """Run the Kalman filter and return the two filtered components.

    Parameters
    ----------
    X : numpy.ndarray
        Full log-price series (train + test).
    params : sequence
        Calibrated ``(a, m, sigma_c, sigma_l)``.
    h : float
        Time step.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The short-term ``c_hat`` and long-term ``l_hat`` series.
    """
    alpha, _, _ = kalman_filter(X, params, h)
    return alpha[:, 0], alpha[:, 1]


# ---------------------------------------------------------------------------
# XGBoost hybrid
# ---------------------------------------------------------------------------

def xgboost_hybrid(c_series, l_series, train_frac=0.8):
    """Train component-wise XGBoost models and recombine their forecasts.

    A shallow model (``max_depth=3``) is used for the fast short-term
    component and a slightly deeper one (``max_depth=4``, lower learning
    rate) for the slow long-term trend.

    Parameters
    ----------
    c_series, l_series : pandas.Series
        Kalman components, date-indexed (typically the filtered train set).
    train_frac : float
        Chronological train fraction.

    Returns
    -------
    dict
        ``pred`` (recombined forecast), ``y_true`` (reconstructed log-price),
        ``pred_c``, ``pred_l``, and the two fitted models.
    """
    feat_c = build_features_short(c_series, n_lags=3)
    feat_l = build_features_long(l_series)

    cols_c = [c for c in feat_c.columns if c != "y"]
    cols_l = [c for c in feat_l.columns if c != "y"]

    n_c = int(len(feat_c) * train_frac)
    n_l = int(len(feat_l) * train_frac)

    Xc_tr, yc_tr = feat_c[cols_c].iloc[:n_c], feat_c["y"].iloc[:n_c]
    Xc_te, yc_te = feat_c[cols_c].iloc[n_c:], feat_c["y"].iloc[n_c:]
    Xl_tr, yl_tr = feat_l[cols_l].iloc[:n_l], feat_l["y"].iloc[:n_l]
    Xl_te, yl_te = feat_l[cols_l].iloc[n_l:], feat_l["y"].iloc[n_l:]

    model_c = train_xgboost(Xc_tr, yc_tr, max_depth=3)
    model_l = train_xgboost(Xl_tr, yl_tr, max_depth=4, learning_rate=0.03)

    pred_c = model_c.predict(Xc_te)
    pred_l = model_l.predict(Xl_te)

    n = min(len(pred_c), len(pred_l))
    pred = pred_c[:n] + pred_l[:n]
    y_true = yc_te.values[:n] + yl_te.values[:n]

    return {"pred": pred, "y_true": y_true,
            "pred_c": pred_c[:n], "pred_l": pred_l[:n],
            "model_c": model_c, "model_l": model_l}


# ---------------------------------------------------------------------------
# LSTM hybrid
# ---------------------------------------------------------------------------

def lstm_hybrid(c_train, l_train, c_test, l_test, T=7, device="cpu"):
    """Train component-wise LSTMs and recombine their forecasts.

    Each component is scaled and modeled by its own LSTM; the recombined
    forecast is ``c_hat + l_hat`` on the test set.

    Parameters
    ----------
    c_train, l_train : numpy.ndarray
        Filtered components on the train set.
    c_test, l_test : numpy.ndarray
        Filtered components on the test set.
    T : int
        Sequence length.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict
        ``pred`` (recombined), ``pred_c``, ``pred_l``, ``y_true``.
    """
    model_c, scaler_c = train_lstm(c_train, T=T, device=device)
    model_l, scaler_l = train_lstm(l_train, T=T, device=device)

    pred_c = lstm_predict(model_c, c_test, scaler_c, T=T, device=device)
    pred_l = lstm_predict(model_l, l_test, scaler_l, T=T, device=device)

    n = min(len(pred_c), len(pred_l))
    pred = pred_c[:n] + pred_l[:n]
    y_true = c_test[T:][:n] + l_test[T:][:n]

    return {"pred": pred, "y_true": y_true,
            "pred_c": pred_c[:n], "pred_l": pred_l[:n]}
