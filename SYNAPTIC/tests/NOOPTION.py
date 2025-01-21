import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import backtrader as bt
import sys
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

# Add the parent directory to sys.path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

def get_optimal_portfolio(df_window):
    mu = mean_historical_return(df_window)
    S = CovarianceShrinkage(df_window).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    # ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    risk_free_rate = mu.min() # for now risk_free_rate is the minimum return of the all returns - must be changed not sure how
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    ticker_list = df_window.columns.tolist()
    var_names = ["Time_Start", "Time_End"] + ticker_list
    w = pd.DataFrame(columns=var_names, index=range(1))
    w["Time_Start"] = df_window.index[0]
    w["Time_End"] = df_window.index[-1]
    w[ticker_list] = list(cleaned_weights.values())

    return w

def get_raw_prices(ticker_list, start_date, end_date):
    url = 'https://api.orats.io/datav2/hist/dailies'
    payload = {
        'token': API_KEY,
        'ticker': ','.join(ticker_list),
        'tradeDate': start_date + ',' + end_date,
        'fields': 'tradeDate,ticker,clsPx'
    }

    response = requests.get(url, params=payload)
    response_dict = response.json()

    # Extracting 'data' from JSON
    data_list = response_dict['data']

    # Creating DataFrame
    ticker_df = pd.DataFrame(data_list)

    # Reformat
    ticker_df = ticker_df[['tradeDate', 'ticker', 'clsPx']]
    # Convert 'date' column to datetime
    ticker_df['tradeDate'] = pd.to_datetime(ticker_df['tradeDate'])
    ticker_df = ticker_df.reset_index().rename(columns={'tradeDate': 'Time'})
    ticker_df.set_index('Time', inplace=True)
    ticker_df = ticker_df.pivot(columns='ticker', values='clsPx')
    ticker_df.columns.name = None

    return ticker_df

def get_all_data(ticker_list, start_date, end_date):
        url = 'https://api.orats.io/datav2/hist/dailies'
        payload = {
            'token': API_KEY,
            'ticker': ','.join(ticker_list),
            'tradeDate': start_date + ',' + end_date,
            'fields': 'tradeDate,ticker,open,hiPx,loPx,clsPx,stockVolume'
        }

        response = requests.get(url, params=payload)
        response_dict = response.json()

        # Extracting 'data' from JSON
        data_list = response_dict['data']

        # Creating DataFrame
        ticker_df = pd.DataFrame(data_list)

        # Reformatting the DataFrame
        # Ensure the necessary columns are present
        ticker_df = ticker_df[['tradeDate', 'ticker', 'open', 'hiPx', 'loPx', 'clsPx', 'stockVolume']]

        # Convert 'tradeDate' to datetime and set as index
        ticker_df['tradeDate'] = pd.to_datetime(ticker_df['tradeDate'])
        ticker_df.set_index('tradeDate', inplace=True)

        # Pivot the DataFrame so that tickers are the columns and price types are MultiIndex columns
        price_data = ticker_df.pivot_table(index='tradeDate', columns='ticker', values=['open', 'hiPx', 'loPx', 'clsPx', 'stockVolume'])
        # Rename the columns to match the 'Open', 'High', 'Low', 'Close' format expected by the backtesting library
        rename_dict = {
            'open': 'Open',
            'hiPx': 'High',
            'loPx': 'Low',
            'clsPx': 'Close',
            'stockVolume': 'Volume'
        }

         # Flatten the MultiIndex columns and apply renaming
        price_data.columns = ['_'.join([ticker, rename_dict[col]]) for col, ticker in price_data.columns]
        
        return price_data


def sliding(ticker_list, start_date, end_date, K):   
    '''
    
    Parameters:
    ticker_list (list of tickers): A list of tickers representing the input data.
    start_date: start date
    end_date: end date
    K (int): The size of the sliding window.
    
    Returns:
    list of int: A list containing the maximum values from each sliding window of size K as it moves through the list arr
    '''
    
    url = 'https://api.orats.io/datav2/hist/dailies'
    payload = {
        'token': API_KEY,
        'ticker': ','.join(ticker_list),
        'tradeDate': start_date + ',' + end_date,
        'fields': 'tradeDate,ticker,clsPx'
    }

    response = requests.get(url, params=payload)
    response_dict = response.json()

    # Extracting 'data' from JSON
    data_list = response_dict['data']

    # Creating DataFrame
    ticker_df = pd.DataFrame(data_list)

    # Reformat
    ticker_df = ticker_df[['tradeDate', 'ticker', 'clsPx']]
    # Convert 'date' column to datetime
    ticker_df['tradeDate'] = pd.to_datetime(ticker_df['tradeDate'])
    ticker_df.set_index('tradeDate', inplace=True)
    ticker_df = ticker_df.reset_index().rename(columns={'tradeDate': 'Time'}).set_index('Time')
    ticker_df = ticker_df.pivot(columns='ticker', values='clsPx')
    ticker_df.columns.name = None
    
    var_names = ["Time_Start", "Time_End"] + ticker_df.columns.tolist()

    result_df = pd.DataFrame(columns=var_names)
    result_df["Time_Start"] = ticker_df.index
    result_df["Time_End"] = result_df["Time_Start"].shift(-K) #personal note: shift up by K instead of adding K days bc you not every day has price
    result_df = result_df.iloc[:len(ticker_df) - K] # take out the last K results bc out of range

    #get the optimal portfolio for every window, 
    for i, row in result_df.iterrows():
        start_time = row["Time_Start"]
        end_time = row["Time_End"]

        window = ticker_df.loc[start_time:end_time]
        # Lookback function
        weights = get_optimal_portfolio(window)
        
        result_df.loc[i, ticker_df.columns] = weights[ticker_df.columns].values

    # Set Time_Start as the index
    result_df.set_index(['Time_End'], inplace=True) #index is time end because we are "looking back" in the window to determine weights

    return result_df

def train_model(ticker_list, start_date, end_date, K, futureSteps):
    
    sliding_data = sliding(ticker_list, start_date, end_date, K)
    
    Xtrain = np.array([sliding_data[ticker].apply(lambda x: x[0]) for ticker in ticker_list]).T
    
    # Ytrain: target values: Xtrain vals shifted by futureSteps upward
    Ytrain = np.roll(Xtrain, -futureSteps, axis=0)
    
    #slice, remove values that were rolled around
    Xtrain = Xtrain[:-futureSteps, :]
    Ytrain = Ytrain[:-futureSteps, :]

    #train model
    models = []
    for i in range(Ytrain.shape[1]):
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        model.fit(Xtrain, Ytrain[:, i])
        models.append(model)

    return models

def predict_weights(models, ticker_list, start_date, end_date, K):
    #new sliding data
    new_sliding_data = sliding(ticker_list, start_date, end_date, K)
    #turn list into array for each ticker
    Xtest = np.array([new_sliding_data[ticker].apply(lambda x: x[0]) for ticker in ticker_list]).T

    # Generate predictions using the models
    predictions = []
    for model in models:
        pred = model.predict(Xtest)
        predictions.append(pred)

    weights_df = pd.DataFrame(np.column_stack(predictions), columns=[f'{ticker}_weight' for ticker in ticker_list])
    weights_df.index = new_sliding_data.index #ensure there are dates

    #normalize, divides each weight by the total sum of weights for that row, summing to 1
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)
    return weights_df

class MultiTickerStrategy(bt.Strategy):
    params = (('weights_df', None),)  # Pass the DataFrame of weights as a parameter

    def __init__(self):
        self.weights_df = self.params.weights_df  # Store the DataFrame of weights
        self.weights_df.index = pd.to_datetime(self.weights_df.index).date
        
        self.equity = []  # List to store portfolio value over time

    def next(self):
        # Get the current date from the data feed
        current_date = self.data.datetime.date(0)

        # Append current portfolio value to the equity tracker
        self.equity.append(self.broker.getvalue())

        # Check if the current date is in the weights DataFrame
        if current_date in self.weights_df.index:
            # Get the row of weights for the current date
            current_weights = self.weights_df.loc[current_date]

            # Loop through each ticker and allocate based on the current weights
            for data in self.datas:
                ticker = data._name  # Get the ticker name
                column_name = f'{ticker}_weight'  # Match the column in the weights DataFrame

                # Ensure the column exists in the weights DataFrame
                if column_name in current_weights:
                    weight = current_weights[column_name]
                else:
                    weight = 0  # If no weight is defined for this ticker, set to 0

                # Calculate target position size based on portfolio value and weight
                portfolio_value = self.broker.getvalue()  # Total portfolio value in dollars
                target_value = portfolio_value * weight   # Target value in dollars for this asset
                target_size = target_value / data.close[0]  # Target size in number of shares

                # Get current position size in shares
                current_position_size = self.getposition(data).size

                # Buy or sell based on the difference between target and current position size
                if current_position_size < target_size:
                    # Buy to reach the target size
                    size_to_buy = target_size - current_position_size
                    if size_to_buy > 0:  # Avoid negative or zero-sized orders
                        self.buy(data=data, size=size_to_buy)
                        

                elif current_position_size > target_size:
                    # Sell to reduce to the target size
                    size_to_sell = current_position_size - target_size
                    if size_to_sell > 0:  # Avoid negative or zero-sized orders
                        self.sell(data=data, size=size_to_sell)
                        

class BuyAndHoldSP500(bt.Strategy):
    def __init__(self):
        self.buy_executed = False
        self.equity = []  # Initialize an empty list to track portfolio value

    def next(self):
        # Execute the buy order once, allocating all available cash to S&P 500
        if not self.buy_executed:
            available_cash = self.broker.getcash()
            s_and_p_price = self.data.close[0]
            
            # Calculate how many whole units of the S&P 500 to buy
            size = int(available_cash / s_and_p_price)  # ensure we are buying only whole shares (frac shares don't work)

            if size > 0:
                # Place the buy order
                self.buy(size=size)

                #One time execution because we are buying once and holding
                self.buy_executed = True
            
            else: #error handling if not enough money
                print("Not enough cash to buy any shares.")

        # Append the current portfolio value (equity) to the list
        self.equity.append(self.broker.getvalue())

class EqualWeightBuyAndHold(bt.Strategy):
    def __init__(self):
        self.num_stocks = len(self.datas)  # Number of stocks in the ticker list
        self.buy_size = 1.0 / self.num_stocks  # Equal weight for each stock
        self.equity = []

    def start(self):
        """Runs once at the start of the backtest"""
        self.cash_start = self.broker.cash

    def next(self):
        # Only buy once on the first trading day
        
        for i, data in enumerate(self.datas):
            # If no current position in the stock, buy
            if self.getposition(data).size == 0:
                size = self.broker.getcash() * self.buy_size / data.close[0]
                self.buy(data=data, size=size)
        
        # Record the portfolio value over time
        self.equity.append(self.broker.getvalue())

def main():
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'AMD', 'NVDA', 'INTC', 'QCOM', 'IBM']

    today = datetime.today()

    # Set training and testing date ranges
    train_start_date = '2004-01-01'
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

if __name__ == '__main__':
    main()

# notes:
#     tried to work with backtest library but only handles one ticker at a time
#     working with backtrader. biggest problem throughout this whole project is reformatting the dataframe multiple times
#         each of the 3 methods required a different format
#     finally, compared dynamic strategy to a buy and hold strategy for the S&P 500
#   further steps: time based rebalancing or threshold based rebalancing vs buying and selling everyday

#add options data
#truncate data if no data available
#compare to simple buy and hold strategy with even weights for all tickers
#add min, max, sliding window



