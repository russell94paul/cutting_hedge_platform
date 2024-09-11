import pandas as pd
import pandas_ta as ta

def add_sma(df, column='Adj Close', length=10):
    """
    Add Simple Moving Average (SMA) to the dataframe.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing OHLCV data.
    column (str): Column name to calculate SMA on.
    length (int): Window length for SMA.
    
    Returns:
    pd.DataFrame: Dataframe with added SMA.
    """
    df[f'SMA_{length}'] = ta.sma(df[column], length=length)
    return df

def add_rsi(df, column='Adj Close', length=14):
    """
    Add Relative Strength Index (RSI) to the dataframe.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing OHLCV data.
    column (str): Column name to calculate RSI on.
    length (int): Window length for RSI.
    
    Returns:
    pd.DataFrame: Dataframe with added RSI.
    """
    df['RSI'] = ta.rsi(df[column], length=length)
    return df

def add_macd(df, column='Adj Close', fast=12, slow=26, signal=9):
    """
    Add Moving Average Convergence Divergence (MACD) to the dataframe.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing OHLCV data.
    column (str): Column name to calculate MACD on.
    fast (int): Fast period for MACD.
    slow (int): Slow period for MACD.
    signal (int): Signal period for MACD.
    
    Returns:
    pd.DataFrame: Dataframe with added MACD.
    """
    macd = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    return df