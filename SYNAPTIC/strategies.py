import backtrader as bt
import pandas as pd

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