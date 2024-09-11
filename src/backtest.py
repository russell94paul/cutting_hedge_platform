import pandas as pd
import vectorbt as vbt

def calculate_strategy_returns(df, prediction_column='Predicted Return', return_column='Daily Return'):
    """
    Calculate strategy returns based on predicted returns and actual returns.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing predicted returns and actual returns.
    prediction_column (str): Column name for predicted returns.
    return_column (str): Column name for actual returns.
    
    Returns:
    pd.DataFrame: Dataframe with added strategy returns.
    """
    df['Strategy Return'] = df[prediction_column] * df[return_column]
    return df

def calculate_cumulative_returns(df, strategy_return_column='Strategy Return'):
    """
    Calculate cumulative returns for the strategy.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing strategy returns.
    strategy_return_column (str): Column name for strategy returns.
    
    Returns:
    pd.Series: Cumulative returns.
    """
    cumulative_return = (1 + df[strategy_return_column]).cumprod() - 1
    return cumulative_return

def backtest_strategy(df, prediction_column='Predicted Return', return_column='Daily Return'):
    """
    Perform a comprehensive backtest for the strategy.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing predicted returns and actual returns.
    prediction_column (str): Column name for predicted returns.
    return_column (str): Column name for actual returns.
    
    Returns:
    pd.DataFrame: Dataframe with added strategy returns and cumulative returns.
    """
    df = calculate_strategy_returns(df, prediction_column, return_column)
    df['Cumulative Return'] = calculate_cumulative_returns(df)
    return df

def convert_to_signals(df, prediction_column='Predicted Return', threshold=0):
    """
    Convert predicted returns to buy/sell signals based on a threshold.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing predicted returns.
    prediction_column (str): Column name for predicted returns.
    threshold (float): Threshold for generating buy/sell signals.
    
    Returns:
    pd.DataFrame: Dataframe with added signals.
    """
    df['Signal'] = df[prediction_column].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    return df

def vectorbt_backtest(df, price_column='Adj Close', signal_column='Signal'):
    """
    Perform backtesting using vectorbt.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing price data and signals.
    price_column (str): Column name for price data.
    signal_column (str): Column name for trading signals.
    
    Returns:
    vbt.Portfolio: vectorbt Portfolio object with backtest results.
    """
    # Create entries and exits based on signals
    entries = df[signal_column] == 1
    exits = df[signal_column] == -1
    
    # Perform backtest using vectorbt
    portfolio = vbt.Portfolio.from_signals(df[price_column], entries, exits)
    
    return portfolio