"""
Machine learning forecasters: XGBoost and LSTM.

Both operate on the same shifted log-price series as the Schwartz models.
XGBoost builds prediction intervals via quantile regression; the LSTM uses
Monte Carlo Dropout. Feature engineering helpers for the raw series and for
the Kalman components (used by the hybrid models) live here too.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(series, n_lags=7):
    """Build lag / calendar / rolling features from a date-indexed series."""
    df = pd.DataFrame(index=series.index)
    df["y"] = series.values
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    df["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    df["rolling_std_7"] = series.shift(1).rolling(7).std()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df.dropna()


def build_features_short(series, n_lags=3):
    """Features tuned for the short-term component ``c_t`` (fast dynamics)."""
    df = pd.DataFrame(index=series.index)
    df["y"] = series.values
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["month"] = df.index.month
    df["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    df["ecart_mean"] = series.shift(1) - series.shift(1).rolling(7).mean()
    return df.dropna()


def build_features_long(series):
    """Features tuned for the long-term component ``l_t`` (slow trend)."""
    df = pd.DataFrame(index=series.index)
    df["y"] = series.values
    for lag in [1, 3, 7, 14]:
        df[f"lag_{lag}"] = series.shift(lag)
    df["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = series.shift(1).rolling(14).mean()
    df["rolling_std_7"] = series.shift(1).rolling(7).std()
    df["trend"] = (series.shift(1).rolling(7).mean()
                   - series.shift(1).rolling(14).mean())
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    return df.dropna()


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def _xgb_regressor(**overrides):
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def train_xgboost(X_train, y_train, **overrides):
    """Fit an XGBoost point forecaster."""
    model = _xgb_regressor(**overrides)
    model.fit(X_train, y_train)
    return model


def train_quantile_model(X_train, y_train, quantile, **overrides):
    """Fit an XGBoost quantile regressor for one interval bound."""
    model = _xgb_regressor(objective="reg:quantileerror",
                           quantile_alpha=quantile, **overrides)
    model.fit(X_train, y_train)
    return model


def xgboost_interval(X_train, y_train, X_test, alpha=0.05):
    """Train low/high quantile models and predict the interval bounds.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Lower and upper bounds for the ``1 - alpha`` interval.
    """
    low = train_quantile_model(X_train, y_train, alpha / 2)
    high = train_quantile_model(X_train, y_train, 1 - alpha / 2)
    return low.predict(X_test), high.predict(X_test)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Two-layer LSTM (64 -> 32) with dropout, for one-step-ahead forecasting.

    Dropout is applied only when ``training=True``, which also enables Monte
    Carlo Dropout at inference time for uncertainty estimation.
    """

    def __init__(self, input_size=1, hidden1=64, hidden2=32,
                 dense_units=16, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(hidden2, dense_units)
        self.relu = nn.ReLU()
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x, training=False):
        out, _ = self.lstm1(x)
        out = self.dropout(out) if training else out
        out, (h, _) = self.lstm2(out)
        h = h.squeeze(0)
        out = self.dropout(h) if training else h
        out = self.relu(self.dense1(out))
        return self.output(out).squeeze(1)


def create_sequences(series, T):
    """Turn a 1D series into overlapping ``(window, next_value)`` pairs."""
    X, y = [], []
    for i in range(len(series) - T):
        X.append(series[i:i + T])
        y.append(series[i + T])
    return np.array(X), np.array(y)


def train_lstm(train_series, T=7, epochs=100, patience=15, batch_size=32,
               lr=1e-3, seed=42, device="cpu"):
    """Scale, sequence, and train an LSTM with early stopping.

    Parameters
    ----------
    train_series : numpy.ndarray
        Training log-price series (unscaled).
    T : int
        Sequence length (validated empirically; 7 is optimal on daily data).
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    (LSTMModel, MinMaxScaler)
        Trained model and the fitted scaler (needed to invert predictions).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    X_seq, y_seq = create_sequences(scaled, T)

    X_tr = torch.FloatTensor(X_seq).unsqueeze(-1).to(device)
    y_tr = torch.FloatTensor(y_seq).to(device)
    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    best_val, patience_c, best_weights = float("inf"), 0, None
    n_val = int(len(X_tr) * 0.1)

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb, training=True), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_tr[-n_val:]), y_tr[-n_val:]).item()

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                break

    model.load_state_dict(best_weights)
    return model, scaler


def lstm_predict(model, test_series, scaler, T=7, device="cpu"):
    """Predict one-step-ahead on the test set and invert the scaling."""
    scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()
    X_seq, _ = create_sequences(scaled, T)
    X_te = torch.FloatTensor(X_seq).unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_te).cpu().numpy()
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


def lstm_mc_dropout(model, test_series, scaler, T=7, n_samples=200,
                    alpha=0.05, device="cpu"):
    """Monte Carlo Dropout prediction interval.

    Runs the model ``n_samples`` times with dropout active and takes the
    empirical percentiles as interval bounds.

    Returns
    -------
    dict
        ``mean``, ``lower``, ``upper`` arrays.
    """
    scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()
    X_seq, _ = create_sequences(scaled, T)
    X_te = torch.FloatTensor(X_seq).unsqueeze(-1).to(device)

    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_te, training=True).cpu().numpy())
    model.eval()

    preds = np.array(preds)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)

    lo = np.percentile(preds, 100 * alpha / 2, axis=0)
    hi = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return {"mean": preds.mean(axis=0), "lower": lo, "upper": hi}
