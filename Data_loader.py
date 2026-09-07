"""
Data loading and preprocessing for ENTSO-E day-ahead electricity prices.

Handles the three ENTSO-E export files (2024 hourly, 2025-2026 15-min),
aggregates them to a common daily resolution, and prepares the shifted
log-price series used by every model.
"""

import numpy as np
import pandas as pd


def load_entso_file(fname, resolution="H"):
    """Load a single ENTSO-E day-ahead price export.

    The ENTSO-E timestamp format is
    ``"01/01/2024 00:00:00 - 01/01/2024 01:00:00 (CET)"``; only the start
    of the interval is kept and the trailing timezone tag is stripped.

    Parameters
    ----------
    fname : str
        Path to the ``.xlsx`` export.
    resolution : str
        Human-readable label of the native resolution (for logging only).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``datetime`` with a single ``price`` column, or an empty
        frame if the file is missing.
    """
    try:
        tmp = pd.read_excel(
            fname, skiprows=7, header=0, usecols=[0, 1], names=["MTU", "price"]
        )
        start_str = (
            tmp["MTU"]
            .str.split(" - ").str[0]
            .str.strip()
            .str.replace(r"\s*\(.*\)", "", regex=True)
            .str.strip()
        )
        tmp["datetime"] = pd.to_datetime(start_str, format="%d/%m/%Y %H:%M:%S")
        tmp = tmp[["datetime", "price"]].copy()
        tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
        tmp = tmp.dropna().set_index("datetime").sort_index()
        print(f"{fname}: {len(tmp)} observations ({resolution})")
        return tmp
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return pd.DataFrame()


def load_daily_prices(files=None, data_dir="data"):
    """Load and aggregate all ENTSO-E files to a daily price series.

    Parameters
    ----------
    files : list of str, optional
        File names to load. Defaults to the three project files.
    data_dir : str
        Directory containing the files.

    Returns
    -------
    pandas.DataFrame
        Daily prices indexed by date with a single ``price`` column.
    """
    if files is None:
        files = ["prices_2024.xlsx", "prices_2025.xlsx", "prices_2026.xlsx"]

    resolutions = ["H", "15min", "15min"]
    daily_series = []
    for fname, res in zip(files, resolutions):
        df = load_entso_file(f"{data_dir}/{fname}", resolution=res)
        if not df.empty:
            daily_series.append(df["price"].resample("D").mean())

    if not daily_series:
        raise FileNotFoundError(
            "No ENTSO-E files found. See data/README.md for instructions."
        )

    df_daily = pd.concat(daily_series).to_frame()
    df_daily = df_daily.dropna().sort_index()
    df_daily.columns = ["price"]

    n_neg = (df_daily["price"] < 0).sum()
    print(f"\nTotal after aggregation: {len(df_daily)} days")
    print(f"Period: {df_daily.index.min().date()} -> {df_daily.index.max().date()}")
    print(f"Negative prices: {n_neg} days ({n_neg / len(df_daily) * 100:.1f}%)")
    return df_daily


def compute_shift(prices):
    """Return the shift applied before the log transform.

    ``shift = |min(price)| + 1`` guarantees strictly positive values even
    with negative prices, so the log is always defined.
    """
    p_min = prices.min()
    return abs(p_min) + 1 if p_min <= 0 else 0.0


def to_log_price(prices, shift=None):
    """Transform prices to shifted log-prices ``X_t = ln(S_t + shift)``.

    Returns
    -------
    (numpy.ndarray, float)
        The log-price array and the shift that was used.
    """
    if shift is None:
        shift = compute_shift(prices)
    return np.log(np.asarray(prices) + shift), shift


def train_test_split(series, train_frac=0.8):
    """Chronological split (no shuffling) preserving temporal order.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Train and test arrays.
    """
    arr = np.asarray(series)
    n_train = int(len(arr) * train_frac)
    return arr[:n_train], arr[n_train:]
