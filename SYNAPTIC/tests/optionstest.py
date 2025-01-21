
import webbrowser
import sys
import os

# Add the parent directory to sys.path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import *

from datetime import datetime, timedelta

def display(df, file_name='output.html'):
    # Save the DataFrame as an HTML file
    df.to_html(file_name)

    # Open the saved HTML file in a new browser tab
    webbrowser.open_new_tab(file_name)

# ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'AMD', 'NVDA', 'INTC', 'QCOM', 'IBM']
ticker_list = ['MSFT', 'NVDA', 'GOOGL']

today = datetime.today() 

# Set training and testing date ranges
train_start_date = '2019-01-03'
train_end_date = (today - timedelta(days= 365 + 365 + 365)).strftime('%Y-%m-%d')  # End training 3 years from now


K = 8  # Size of sliding window

# df_calloption_volume = get_calloption_volume(ticker_list, train_start_date, train_end_date)
df_sliding = sliding(ticker_list,train_start_date,train_end_date,K)
# df_sliding_minmax = sliding_minmax(ticker_list,train_start_date,train_end_date,K)

# df_calloption_volume.to_csv('calloption_volume_output.csv', index=True)
# Xtrain_weights =  pd.DataFrame({ticker: df_sliding[ticker].apply(lambda x: x[0]) for ticker in ticker_list})
df_sliding.to_csv('sliding_output.csv', index=True)
# Xtrain_weights.to_csv('new_sliding_output.csv', index=True)
# df_sliding_minmax.to_csv('sliding_minmax_output.csv', index=True)

print('complete')
