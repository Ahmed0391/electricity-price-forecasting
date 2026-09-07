"""
Evaluation metrics for point forecasts and prediction intervals.

Point accuracy is measured with MAE and RMSE; interval quality with the
empirical coverage and mean width. A well-calibrated 95% interval should
cover close to 95% of observations while staying as narrow as possible.
"""

import numpy as np
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox


def mae(y_true, y_pred):
    """Mean absolute error (robust to extreme values)."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error (penalizes large errors)."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def interval_coverage(y_true, lower, upper):
    """Empirical coverage: fraction of observations inside the interval (%)."""
    y_true = np.asarray(y_true)
    return np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))) * 100


def interval_width(lower, upper):
    """Mean width of the prediction interval."""
    return np.mean(np.asarray(upper) - np.asarray(lower))


def residual_tests(residuals, label=""):
    """Normality (Jarque-Bera) and autocorrelation (Ljung-Box) tests.

    Returns
    -------
    dict
        ``skewness``, ``kurtosis`` (raw, so 3 = normal), and the two p-values.
        A rejection of both confirms the residuals are not Gaussian white
        noise -- the expected outcome on real electricity data.
    """
    residuals = np.asarray(residuals)
    _, jb_pvalue = jarque_bera(residuals)
    lb = acorr_ljungbox(residuals, lags=[max(len(residuals) // 3, 1)],
                        return_df=True)
    return {"label": label,
            "skewness": skew(residuals),
            "kurtosis": kurtosis(residuals) + 3,
            "jb_pvalue": jb_pvalue,
            "bp_pvalue": lb["lb_pvalue"].values[0]}


def evaluate(y_true, y_pred, lower=None, upper=None, label=""):
    """Bundle point and (optional) interval metrics into one dict."""
    out = {"label": label, "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}
    if lower is not None and upper is not None:
        out["coverage"] = interval_coverage(y_true, lower, upper)
        out["width"] = interval_width(lower, upper)
    return out
