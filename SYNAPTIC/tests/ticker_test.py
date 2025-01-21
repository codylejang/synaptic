import sys
import os
import requests
from datetime import datetime, timedelta

# Add the parent directory to sys.path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TICKER_LIST,API_KEY

def check_ticker(ticker_list, start_date, end_date):
    url = 'https://api.orats.io/datav2/hist/dailies'

    problematic_tickers = []
    
    for ticker in TICKER_LIST:
        payload = {
            'token': API_KEY,
            'ticker': ticker,
            'tradeDate': start_date + ',' + end_date,
            'fields': 'tradeDate,ticker,clsPx'
        }
        try:
            response = requests.get(url, params=payload)
            response_dict = response.json()

            if 'data' not in response_dict or not response_dict['data']:
                print(f"No data found for ticker: {ticker}")
                problematic_tickers.append(ticker)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for ticker: {ticker}. Error: {e}")
            problematic_tickers.append(ticker)

        except Exception as e:
            print(f"Unexpected error for ticker: {ticker}. Error: {e}")
            problematic_tickers.append(ticker)

    return problematic_tickers

today = datetime.today()
# start_date = (today - timedelta(days=365 + 365)).strftime('%Y-%m-%d')      # Start testing 2 years from now
start_date = '2021-06-01'
end_date = today.strftime('%Y-%m-%d')                                 # End testing today

print(check_ticker(TICKER_LIST,start_date,end_date))

