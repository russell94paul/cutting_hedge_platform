import pandas as pd


def preprocess_ohlcv_data(ohlcv_data):
    # Convert the index to datetime if it's not already
    ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
    
    # Sort the dataframe by date
    ohlcv_data = ohlcv_data.sort_index()
    
    # Fill any missing values (if any) with the forward fill method
    ohlcv_data = ohlcv_data.fillna(method='ffill')
    
    # Calculate additional features (e.g., daily returns)
    ohlcv_data['Daily Return'] = ohlcv_data['Adj Close'].pct_change()
    
    # Fill the NaN value in the first row with 0
    ohlcv_data['Daily Return'] = ohlcv_data['Daily Return'].fillna(0)
    
    return ohlcv_data
