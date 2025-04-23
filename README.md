
# Dynamic Portfolio Optimization with Sliding Windows and Backtesting

This project implements a dynamic portfolio allocation strategy using machine learning and backtesting. It combines predictive modeling, optimal portfolio construction with PyPortfolioOpt, and comparative analysis with traditional investment strategies like equal weighting and S&P 500 buy-and-hold.

## Project Overview

This system predicts future asset allocation weights for a portfolio of tickers using a sliding window approach, leveraging historical price and options data. The predicted weights are used in a backtesting engine powered by Backtrader to simulate real-world performance and compare it against benchmark strategies.

## Key Features

- Sliding Window Forecasting: Uses `K`-day windows to train a regression model predicting weights `FUTURE_STEPS` into the future.
- Options Data Integration: Pulls call volume and beta values from the ORATS API.
- Gradient Boosting Regressor: Models trained per ticker to forecast optimal weights.
- Portfolio Optimization: Implements Efficient Frontier optimization via PyPortfolioOpt for max Sharpe ratio allocation.
- Backtesting Engine: Evaluates strategies using Backtrader.
- Strategy Comparison:
  - Dynamic Strategy (ML-based predictions)
  - Equal Weight Buy-and-Hold
  - S&P 500 Buy-and-Hold

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Runs the training and prediction pipeline and outputs weights to HTML |
| `functions.py` | Core ML logic, data fetching from ORATS, optimization methods |
| `strategies.py` | Backtrader strategies for dynamic weighting, equal weight, and SP500 |
| `backtest.py` | Backtests the predicted weights and plots performance vs. benchmarks |
| `config.py` | Contains constants like API key, ticker list, K, and prediction horizon |
| `weights_output.html` | HTML output of predicted weights (for inspection) |

### Subfolder: `tests/`

| File | Description |
|------|-------------|
| `tests/NOOPTION.py` | Version of model that omits options data (uses prices only) |
| `tests/ticker_test.py` | Checks for problematic or inactive tickers from ORATS |

## Data Sources

- ORATS API for:
  - Daily closing prices
  - Call option volumes
  - 1-month rolling beta
- Yahoo Finance for:
  - S&P 500 (^GSPC) benchmark data (via `yfinance`)

## Modeling Pipeline

1. Train Models: `GradientBoostingRegressor` trained using lagged data for each ticker
2. Predict Future Weights: Predict portfolio weights `FUTURE_STEPS` into the future
3. Backtest Predictions: Use Backtrader to simulate trades and equity curve
4. Compare Benchmarks: Assess ML strategy vs. traditional investing

## Backtesting Metrics

- Sharpe Ratio
- Max Drawdown
- Cumulative Return
- Portfolio Value Over Time

These are calculated and visualized after backtesting in `backtest.py`.

## Requirements

```bash
pip install pandas numpy scikit-learn yfinance matplotlib backtrader pypfopt requests
```

## Author

**Cody Lejang**  
B.S. in Cognitive Science, Specialization in Computing, minor in Data Science Engineering – UCLA  
Interested in the intersection of machine learning, psychology, and data analytics.

## Getting Started

1. Add your ORATS API key in `config.py`
2. Run:

```bash
python main.py
```

3. Review the weight predictions (`weights_output.html`)
4. Run:

```bash
python backtest.py
```

to evaluate performance vs. benchmarks
