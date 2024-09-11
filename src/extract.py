import yfinance as yf
import os

def extract_ohlcv_data(ticker, start_date, end_date, file_path):
    """
    Extract OHLCV data from yfinance API and save it as a parquet file.
    
    Parameters:
    ticker (str): Ticker symbol of the stock.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    end_date (str): End date in the format 'YYYY-MM-DD'.
    file_path (str): Path to save the parquet file.
    
    Returns:
    None
    """
    # Download data from yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save data as a parquet file
    data.to_parquet(file_path)
    
    return data
    
 
#ticker = 'AMZN' 
#start_date = '2020-01-01'
#end_date = '2021-01-01'
#file_path = f'data/raw/{ticker}_ohlcv.parquet'
#extract_ohlcv_data(ticker,  start_date, end_date, file_path)  


# TO-DO: Add data extract from API - MT5  


# TO-DO: Add data extract from API - Binance