# electricity-price-forecasting
Electricity spot price forecasting using stochastic models and ML/DL approaches
# Electricity Spot Price Forecasting in the Dutch Market 

Hybrid stochastic + machine learning approach for forecasting day-ahead
electricity spot prices on the Dutch market, combining Schwartz factor
ML models.

Research internship at LAMSIN, ENIT / ENSTA Paris (2026)

---

## Overview

Electricity spot prices are hard to model: strong multi-scale seasonality,
mean reversion, sudden spikes, and  on modern markets  negative prices
driven by renewable overproduction. This project compares six models for
day-ahead price forecasting and proposes an original **hybrid Kalman + ML**
approach.

## Key Contribution

The core contribution is a hybrid method that uses the **Kalman filter** to
decompose the log-price into a short-term mean-reverting component (`c_t`,
~81% of variance) and a long-term random-walk component (`l_t`, ~19%), then
trains **specialized ML models on each component** before recombining.

This decomposition systematically improves performance, regardless of the
ML back-end used (−12.8% MAE for XGBoost, −14.2% for LSTM).

## Results

Test set (181 days), metrics in log-price space:

| Model | MAE | RMSE | PI Coverage | PI Width |
|-------|-----|------|-------------|----------|
| Schwartz 1-factor | 0.295 | 0.384 | 98.3% | 1.726 |
| XGBoost classic | 0.241 | 0.357 | 82.8% | 0.769 |
| **XGBoost hybrid** | **0.210** | **0.334** | **94.4%** | 0.998 |
| LSTM classic | 0.249 | 0.365 | 86.8% | 0.795 |
| LSTM hybrid | 0.231 | 0.359 | 90.0% | 0.870 |

**XGBoost hybrid** offers the best trade-off between point accuracy and
interval calibration. No single model dominates on every criterion.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py   # ENTSO-E loading + preprocessing
│   ├── schwartz.py      # 1F/2F models + Kalman filter
│   ├── ml_models.py     # XGBoost + LSTM
│   ├── hybrid.py        # hybrid Kalman + ML (main contribution)
│   └── metrics.py       # MAE, RMSE, interval coverage/width
├── notebooks/
│   └── demo.ipynb       # end-to-end demonstration
├── figures/             # exported figures
├── report/
│   └── rapport.pdf      # full report (French)
└── data/
    └── README.md        # ENTSO-E download instructions
```

## Methods

**Stochastic models**
- Ornstein-Uhlenbeck process, exact AR(1) discretization
- OLS / MLE calibration (proven equivalent for the Gaussian AR(1))
- Two-factor state-space model estimated via the Kalman filter
- Simulation-based validation of the estimator

**Machine Learning**
- XGBoost with engineered features (lags, calendar, rolling statistics);
  prediction intervals via quantile regression
- LSTM (2 layers, sequence length validated empirically); prediction
  intervals via Monte Carlo Dropout

**Hybrid**
- Kalman decomposition feeding component-wise ML models

## Getting Started

```bash
git clone https://github.com/Ahmed0391/electricity-price-forecasting.git
cd electricity-price-forecasting
pip install -r requirements.txt
# add ENTSO-E files to data/ (see data/README.md)
jupyter notebook notebooks/demo.ipynb
```

## Tech Stack

Python · PyTorch · XGBoost · scikit-learn · SciPy · pandas · NumPy

## Author

**Ahmed Saidi**, Applied Mathematics student at ENIT-ENSTA Paris
Research internship at LAMSIN, supervised by H. Mezghani


## Acknowledgements

Based on the methodology of Senhadji El Rhazi (2004), *Modélisation et
calibration des prix spot électriques*, Université Paris VI / EDF.
