import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from config import API_KEY, TICKER_LIST
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

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

def get_calloption_volume(ticker_list, start_date, end_date):
    url = 'https://api.orats.io/datav2/hist/cores'
    
    # Function to chunk ticker_list into batches of 4 because for some reason API can't accept many tickers at once
    def chunk_tickers(tickers, chunk_size=4):
        for i in range(0, len(tickers), chunk_size):
            yield tickers[i:i + chunk_size]

    combined_df = pd.DataFrame()
    
    for chunk in chunk_tickers(ticker_list, 4):
        payload = {
            'token': API_KEY,
            'ticker': ','.join(chunk),
            'fields': 'tradeDate,ticker,cVolu,beta1m'
        }

        response = requests.get(url, params=payload)
        response_dict = response.json()

        # Extracting 'data' from JSON
        data_list = response_dict['data']

        # Creating DataFrame
        options_df = pd.DataFrame(data_list)
        
        # Reformat
        options_df = options_df[['tradeDate', 'ticker', 'cVolu', 'beta1m']]
        # Convert 'tradeDate' column to datetime
        options_df['tradeDate'] = pd.to_datetime(options_df['tradeDate'])
        options_df = options_df.reset_index().rename(columns={'tradeDate': 'Time'})
        options_df.set_index('Time', inplace=True)
        options_df = options_df.pivot(columns='ticker', values=['cVolu', 'beta1m'])
        options_df.columns.name = None
        
        # Flatten the multi-level columns to create single-level columns like 'AAPL_cVolu', 'AAPL_beta1m'
        options_df.columns = [f'{ticker}_{metric}' for metric, ticker in options_df.columns]

        combined_df = pd.concat([combined_df, options_df], axis=1)

    combined_df.sort_index(inplace=True)

    reordered_columns = [f'{ticker}_cVolu' for ticker in ticker_list] + [f'{ticker}_beta1m' for ticker in ticker_list]
    combined_df = combined_df[reordered_columns]  # Reorder according to the ticker list

    #filter from start date to end date
    combined_df = combined_df.loc[start_date:end_date]

    return combined_df
    
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

def sliding_minmax(ticker_list, start_date, end_date, K):
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
    
    ticker_df.columns.name = None #get rid of 'ticker' column label

     # Calculate sliding minimum and maximum
    sliding_min = ticker_df.rolling(window=K, min_periods=1).min()
    sliding_max = ticker_df.rolling(window=K, min_periods=1).max()

    var_names = ["Time_Start", "Time_End"] + [f"{col}_min" for col in ticker_df.columns] + [f"{col}_max" for col in ticker_df.columns]

    result_df = pd.DataFrame(columns=var_names)
    result_df["Time_Start"] = ticker_df.index
    result_df["Time_End"] = result_df["Time_Start"].shift(-K) #personal note: shift up by K instead of adding K days bc  not every day has price
    
    for col in ticker_df.columns:
        result_df[f"{col}_min"] = sliding_min[col].values
        result_df[f"{col}_max"] = sliding_max[col].values
    
    result_df = result_df.iloc[:len(ticker_df) - K] # take out the last K results bc out of range

    result_df.set_index(['Time_End'], inplace=True)
    result_df.drop('Time_Start', axis=1, inplace=True)
    
    return result_df

def sliding(ticker_list, start_date, end_date, K):   
    
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
    result_df.drop('Time_Start', axis=1, inplace=True)

    return result_df

def train_model(ticker_list, start_date, end_date, K, futureSteps):
    
    #get weights
    sliding_data = sliding(ticker_list, start_date, end_date, K)
    option_volumes = get_calloption_volume(ticker_list, start_date, end_date)
    minmax_data = sliding_minmax(ticker_list, start_date, end_date, K)
    
    # preprocess weights
    Xtrain_weights = sliding_data
    
    # outer join
    combined_data = pd.concat([Xtrain_weights, option_volumes, minmax_data], axis=1, join="outer")

    # fill missing data
    combined_data.fillna(combined_data.mean(), inplace=True) #mean imputation
    combined_data.fillna(0, inplace=True) # 0 imputation as a safeguard
    
    # slice combined data to match the original training window in case
    combined_data = combined_data.loc[start_date:end_date]

    assert not combined_data.isna().any().any(), "combined_data contains NaN values"
    
    # Separate data after alignment
    Xtrain_weights_aligned = combined_data.iloc[:, :len(ticker_list)].values
    option_volumes_aligned = combined_data.iloc[:, len(ticker_list):len(ticker_list)*3].values  # Adjust based on how many new columns (beta1m) are included
    minmax_data_aligned = combined_data.iloc[:, len(ticker_list)*3:].values  # Adjust for any additional data

    # Combine into a single feature set
    Xtrain = np.hstack([Xtrain_weights_aligned, option_volumes_aligned, minmax_data_aligned])
    Ytrain = np.roll(Xtrain_weights_aligned, -futureSteps, axis=0)  # we only shift stock data for target
    
    #slice, remove values that were rolled around (inaccurate data)
    Xtrain = Xtrain[:-futureSteps, :]
    Ytrain = Ytrain[:-futureSteps, :]

    #train model
    models = []
    for i in range(Ytrain.shape[1]): #each ticker modeled separately
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        model.fit(Xtrain, Ytrain[:, i]) 
        models.append(model)

    return models

def predict_weights(models, ticker_list, start_date, end_date, K):
    #get test data
    new_sliding_data = sliding(ticker_list, start_date, end_date, K)
    Xtest_weights = new_sliding_data
    option_volumes_test = get_calloption_volume(ticker_list, start_date, end_date)
    minmax_test = sliding_minmax(ticker_list, start_date, end_date, K)
    
    # Combine
    combined_test_data = pd.concat([Xtest_weights, option_volumes_test, minmax_test], axis=1, join="outer")
    
    # Fill missing (fix)
    combined_test_data.fillna(combined_test_data.mean(), inplace=True) #mean imputation
    combined_test_data.fillna(0, inplace=True) # 0 imputation as a safeguard

    # ensure dates sorted
    combined_test_data.sort_index(inplace=True)

    # Convert the combined test data into a NumPy array for model prediction
    Xtest = combined_test_data.values

    # Generate predictions using the models
    predictions = []
    for model in models:
        pred = model.predict(Xtest)
        predictions.append(pred)

    weights_df = pd.DataFrame(np.column_stack(predictions), columns=[f'{ticker}_weight' for ticker in ticker_list])
    
    weights_df.index = combined_test_data.index

    #normalize, divides each weight by the total sum of weights for that row, summing to 1
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)
    return weights_df


                        
