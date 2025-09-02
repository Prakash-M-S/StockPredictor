# Stock Price Prediction (yfinance + scikit‑learn)

A reproducible notebook-driven project to download historical stock data via yfinance, engineer simple technical features (SMA7, SMA10, Volume), and train baseline models (Linear Regression, Random Forest) to predict Close prices. Includes quick visualizations (Close, Volume, SMAs) and evaluation (MAE, RMSE, R²), all runnable in a single Jupyter notebook.[2][1]

## Features

- Data loading from yfinance with user-specified ticker and 2-year history.
- Quick EDA: head(), max price, Close/Volume charts.
- Feature engineering: SMA7, SMA10, simple lag structure (as used in notebook), train/test split.
- Models:
  - Linear Regression (baseline) with metrics: MAE, RMSE, R².
  - Random Forest Regressor with feature importances and metrics.
- Plots: Actual vs Predicted for both models; Close with SMAs for last ~6 months.

## Results (current notebook run)

- Linear Regression: MAE ≈ 14.28, RMSE ≈ 17.75, R² ≈ 0.913 (example from notebook output).
- Random Forest: MAE ≈ 21.05, RMSE ≈ 27.25, R² ≈ 0.794 (example from notebook output).
- Random Forest feature importances indicate Close-related features dominate (example array shown in notebook).

Note: Metrics will vary by ticker and date of retrieval due to changing market data.

## Getting started

### Prerequisites

- Python 3.9+ recommended.
- pip or conda environment manager.
- Jupyter or VS Code with Python extension.

### Installation

1) Create and activate a virtual environment
- pip:
  - python -m venv .venv
  - source .venv/bin/activate  (Windows: .venv\Scripts\activate)
- conda:
  - conda create -n stocks python=3.10 -y
  - conda activate stocks[5]

2) Install dependencies
- pip install -U yfinance numpy pandas matplotlib seaborn scikit-learn.

3) Launch Jupyter
- jupyter notebook

4) Open Stock_Prediction-1.ipynb and run all cells.

## Usage

1) When prompted, enter a ticker (e.g., TSLA, AAPL, MSFT). The notebook fetches ~2 years of daily OHLCV from yfinance.
2) The notebook computes SMAs, plots charts, and builds train/test splits.
3) Fit models (Linear Regression and Random Forest), review metrics, and visualize Actual vs Predicted curves.
4) Adjust features (e.g., add more lags, technical indicators), re-run, and compare metrics.

## Reproducibility tips

- Fix a random seed in train/test splitting and models (RandomForestRegressor(random_state=...)) to get stable comparisons across runs.
- Save model artifacts (e.g., joblib dump) and metrics to results/ for later reference if you modularize.
- Record the run date and yfinance version because the data is time-variant.

## Data sources

- yfinance: free Yahoo Finance historical data via API wrapper. License and availability subject to Yahoo terms; data can change or be adjusted retroactively.

## Extensions and roadmap

- Add more features: returns, log returns, RSI, MACD, Bollinger Bands, rolling volatility.
- Time-series aware validation: walk-forward split instead of random split to avoid leakage.
- Hyperparameter tuning: GridSearchCV or Optuna for Random Forest and other models.
- Alternative models: Gradient Boosting (XGBoost/LightGBM), LSTM/Temporal CNN with sequence windows.
- Backtesting framework: event-driven or simple trading simulation to translate predictions to strategies
- Packaging: split into src/ with train.py, predict.py, build_features.py; add config.yml and Makefile for reproducible runs.

