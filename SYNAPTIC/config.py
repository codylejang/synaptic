#config
INITIAL_INVESTMENT = 100000
API_KEY = 'hide'
# TICKER_LIST = ['AAPL', 'ABBV', 'ADBE', 'ADI', 'ADP', 'ADUS', 'AEP', 
#                'ALKS', 'AMGN', 'AMZN', 'APD', 'ATUS',  'ATVI', 'AVGO', 
#                'AXP', 'BA', 'BABA', 'BAC', 'BKNG', 'BMRN', 'CAT', 'CHTR', 
#                'CMCSA', 'COST', 'CSCO',  'CTVA', 'CVX', 'DIS', 'DOW', 'EA', 
#                'EOG', 'EXC', 'FCX', 'FDLO', 'GILD', 'GOOG', 'GOOGL', 'GS', 'HD',  
#                'HON', 'IBM', 'INCY', 'INTC', 'IONS', 'JAZZ', 'JPM', 'KHC', 
#                'KO', 'LLY', 'LOW', 'MA', 'MAR', 'MCD',  'MDLZ', 'MELI', 'META', 
#                'MMM', 'MPC', 'MRVL', 'MSFT', 'MU', 'NFLX',
#                'NKE', 'NUE',  'NVDA', 'NVO', 'NXPI', 'PANW', 'PDBC', 'PEP', 'PFE', 
#                'PG', 'PSA', 'PYPL', 'QCOM', 'REGN', 'RFIL', 'RTX',  'SBUX', 'SHY', 
#                'SLB', 'SLQD', 'SLV', 'SNPS', 'SO', 'SQQQ', 'T', 'TLT', 'TMUS', 
#                'TSLA', 'TSM', 'TTT', 'TXN',  'UNH', 'UTHR', 'UTI', 'V', 'VRSN', 
#                'VTC', 'VYM', 'VZ', 'WBA', 'WMT', 'ZM']
TICKER_LIST = ['AAPL','QCOM','NVDA']

#took out BRLT, DEH bc public in 2021 (not much data)
#NECB, not NEBC in original code
#took out NECB because no options data (will speak soon on how to handle)
#ABLX out bc only available from 2017-2018 (acquired by Sanofi in 2018)
#ALXN out bc acquired by AstraZeneca on July 21, 2021
#JMBA acquired by Focus in September 2018
#MORE acquired by Greystar in September 2017
#NDRO terminated 2018


#solution could be a filter that excludes all problematic tickers before model is trained/tested

K = 8 # Size of sliding window
FUTURE_STEPS = 7 # How many days in the future you want to predict