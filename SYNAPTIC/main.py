
from config import TICKER_LIST, K, FUTURE_STEPS
from datetime import datetime, timedelta
from functions import train_model, predict_weights
import time

def main():
    start = time.time()
    print("started...")

    ticker_list = TICKER_LIST

    today = datetime.today()

    # Set training and testing date ranges
    train_start_date = '2007-01-01'
    train_end_date = (today - timedelta(days= 365 + 365 + 365)).strftime('%Y-%m-%d')  # End training 3 years from now
    
    test_start_date = (today - timedelta(days=365 + 365)).strftime('%Y-%m-%d')      # Start testing (predictions) 2 years from now
    test_end_date = today.strftime('%Y-%m-%d')                                 # End testing (predictions) today

    # Train the model with the specified parameters
    models = train_model(ticker_list, train_start_date, train_end_date, K, FUTURE_STEPS)

    # Predict weights using the trained models and test data
    weights_df = predict_weights(models, ticker_list, test_start_date, test_end_date, K)
    weights_df.to_html("weights_output.html")
    
    end = time.time()
    print(f'Completed in {end-start} seconds')

if __name__ == '__main__':
    main()