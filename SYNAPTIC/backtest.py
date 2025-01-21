import matplotlib as plt
import backtrader as bt
import yfinance as yf
from strategies import MultiTickerStrategy, BuyAndHoldSP500, EqualWeightBuyAndHold
from functions import get_all_data
from config import TICKER_LIST, K
from datetime import datetime, timedelta
from functions import train_model, predict_weights

ticker_list = TICKER_LIST

today = datetime.today()

# Set training and testing date ranges
train_start_date = '2007-01-01'
train_end_date = (today - timedelta(days= 365 + 365 + 365)).strftime('%Y-%m-%d')  # End training 3 years from now

test_start_date = (today - timedelta(days=365 + 365)).strftime('%Y-%m-%d')      # Start testing (predictions) 2 years from now
test_end_date = today.strftime('%Y-%m-%d')                                 # End testing (predictions) today

K = 8  # Size of sliding window
futureSteps = 7  # How many days in the future you want to predict

# Train the model with the specified parameters
models = train_model(ticker_list, train_start_date, train_end_date, K, futureSteps)


# Predict weights using the trained models and test data
weights_df = predict_weights(models, ticker_list, test_start_date, test_end_date, K)
print(weights_df)

# Plotting the predicted weights for each ticker over time
plt.figure(figsize=(14, 7))
for ticker in ticker_list:
    plt.plot(weights_df.index, weights_df[f'{ticker}_weight'], label=f'{ticker} Weight')
plt.title('Predicted Weights Over Time')
plt.legend()
plt.show()

cerebro = bt.Cerebro()

    # Add analyzers to calculate performance metrics
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')  # For cumulative return

# Obtain price data for all tickers
price_data = get_all_data(ticker_list, test_start_date, test_end_date)

# Loop through each ticker to add data to Cerebro
for ticker in ticker_list:
    # Extract the columns for this ticker
    ticker_data = price_data[[f'{ticker}_Open', f'{ticker}_High', f'{ticker}_Low', f'{ticker}_Close', f'{ticker}_Volume']]

    # Rename the columns back to Backtrader's format
    ticker_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Add the data feed to Cerebro
    data_feed = bt.feeds.PandasData(dataname=ticker_data, name=ticker)
    cerebro.adddata(data_feed)

# Add the custom strategy with the defined weights
cerebro.addstrategy(MultiTickerStrategy, weights_df=weights_df)

# Set initial capital
cerebro.broker.set_cash(100000)

# Run the backtest and retrieve the first strategy
strategies = cerebro.run()
dynamic_strategy = strategies[0]

# Plot the results
cerebro.plot()

# Print out analyzer results
print(f"Sharpe Ratio: {dynamic_strategy.analyzers.sharpe.get_analysis()['sharperatio']}")
print(f"Max Drawdown: {dynamic_strategy.analyzers.drawdown.get_analysis().max.drawdown}%")
print(f"Cumulative Return: {dynamic_strategy.analyzers.returns.get_analysis()['rnorm100']}%")

# Plot portfolio performance using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(dynamic_strategy.equity, label='Portfolio Value')
plt.title('Portfolio Performance Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()


# --------------------
# Run S&P 500 Buy-and-Hold Strategy for Comparison
# --------------------

# Reset Cerebro for the S&P 500 strategy
cerebro = bt.Cerebro()

# Download S&P 500 data from Yahoo Finance
sp500_data = yf.download('^GSPC', start=test_start_date, end=test_end_date)

# Prepare S&P 500 data for Backtrader
sp500_data_feed = bt.feeds.PandasData(dataname=sp500_data)
cerebro.adddata(sp500_data_feed)

# Add the S&P 500 buy-and-hold strategy to Cerebro
cerebro.addstrategy(BuyAndHoldSP500)

# Set initial capital for the S&P 500 strategy
cerebro.broker.set_cash(100000)

# Run the backtest for the S&P 500 strategy
strategies = cerebro.run()
sp500_strategy = strategies[0]

# --------------------
# Run Evenly Weighted Buy and Hold for Comparison
# --------------------

cerebro_equal_weight = bt.Cerebro()

# Add tickers data for EqualWeightBuyAndHold strategy
for ticker in ticker_list:
    ticker_data = yf.download(ticker, start=test_start_date, end=test_end_date)
    ticker_data_feed = bt.feeds.PandasData(dataname=ticker_data)
    cerebro_equal_weight.adddata(ticker_data_feed)

# Add EqualWeightBuyAndHold strategy to Cerebro
cerebro_equal_weight.addstrategy(EqualWeightBuyAndHold)

# Set initial capital for Equal Weight strategy
cerebro_equal_weight.broker.set_cash(100000)

equal_weight_strategy = cerebro_equal_weight.run()[0]

# --------------------
# Compare Strategies: Dynamic Strategy vs S&P 500 Buy-and-Hold vs Equal Weight Buy-and-Hold
# --------------------

# Plot portfolio performance for all strategies
plt.figure(figsize=(10, 6))
plt.plot(dynamic_strategy.equity, label='Dynamic Portfolio Value')
plt.plot(sp500_strategy.equity, label='S&P 500 Buy-and-Hold', linestyle='--')
plt.plot(equal_weight_strategy.equity, label='Equal Weight Buy-and-Hold', linestyle='-.')
plt.title('Dynamic Strategy vs S&P 500 Buy-and-Hold vs Equal Weight Buy-and-Hold')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()