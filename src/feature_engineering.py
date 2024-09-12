from feature.feature_library import technical_indicators as ta_ind

def add_technical_indicators(ohlcv):
    """
    Add technical indicators to the dataframe using pandas_ta.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing OHLCV data.
    
    Returns:
    pd.DataFrame: Dataframe with added technical indicators.
    """
    ohlcv = ta_ind.add_sma(ohlcv, length=10)
    ohlcv = ta_ind.add_rsi(ohlcv, length=14)
    ohlcv = ta_ind.add_macd(ohlcv, fast=12, slow=26, signal=9)
    
    # Remove records will null values for indicator columns 
    # expl: depending on the lookback window some feature will not calculated due to no historical price data
    ohlcv.dropna(inplace=True)
    
    # TO-DO: Set first row of data to 0 - CONFIRM IF THIS IS THE CORRECT APPROACH
    
    return ohlcv