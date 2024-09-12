import pandas as pd
import numpy as np

def add_sma(data, window):
    """
    Adds Simple Moving Average (SMA) to the data.
    SMA is the average of the closing prices over a specified period.
    """
    data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def add_ema(data, window):
    """
    Adds Exponential Moving Average (EMA) to the data.
    EMA gives more weight to recent prices, making it more responsive to new information.
    """
    data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
    return data

def add_macd(data, fast_window, slow_window, signal_window):
    """
    Adds Moving Average Convergence Divergence (MACD) to the data.
    MACD is the difference between the 12-day and 26-day EMAs.
    """
    data['MACD'] = data['Close'].ewm(span=fast_window, adjust=False).mean() - data['Close'].ewm(span=slow_window, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def add_rsi(data, window):
    """
    Adds Relative Strength Index (RSI) to the data.
    RSI measures the speed and change of price movements.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    return data

def add_bollinger_bands(data, window, num_std_dev):
    """
    Adds Bollinger Bands to the data.
    Bollinger Bands are volatility bands placed above and below a moving average.
    """
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    data[f'Bollinger_Upper_{window}'] = sma + (std_dev * num_std_dev)
    data[f'Bollinger_Lower_{window}'] = sma - (std_dev * num_std_dev)
    return data

def add_stochastic_oscillator(data, window):
    """
    Adds Stochastic Oscillator to the data.
    Stochastic Oscillator compares a particular closing price to a range of prices over a certain period.
    """
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data[f'Stoch_{window}'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    return data

def add_adx(data, window):
    """
    Adds Average Directional Index (ADX) to the data.
    ADX measures the strength of a trend.
    """
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff()
    tr = np.maximum(data['High'] - data['Low'], np.maximum(abs(data['High'] - data['Close'].shift()), abs(data['Low'] - data['Close'].shift())))
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    data[f'ADX_{window}'] = dx.ewm(alpha=1/window).mean()
    return data

def add_atr(data, window):
    """
    Adds Average True Range (ATR) to the data.
    ATR measures market volatility by decomposing the entire range of an asset price for that period.
    """
    tr = np.maximum(data['High'] - data['Low'], np.maximum(abs(data['High'] - data['Close'].shift()), abs(data['Low'] - data['Close'].shift())))
    data[f'ATR_{window}'] = tr.rolling(window=window).mean()
    return data

def add_cci(data, window):
    """
    Adds Commodity Channel Index (CCI) to the data.
    CCI measures the current price level relative to an average price level over a given period.
    """
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
    data[f'CCI_{window}'] = (tp - sma) / (0.015 * mad)
    return data

def add_roc(data, window):
    """
    Adds Rate of Change (ROC) to the data.
    ROC is a momentum oscillator that measures the percentage change between the current price and the price a certain number of periods ago.
    """
    data[f'ROC_{window}'] = data['Close'].pct_change(periods=window) * 100
    return data

def add_mfi(data, window):
    """
    Adds Money Flow Index (MFI) to the data.
    MFI is a momentum indicator that uses a stock's price and volume to predict the reliability of the current trend.
    """
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    mf = tp * data['Volume']
    pos_mf = mf.where(tp > tp.shift(), 0).rolling(window=window).sum()
    neg_mf = mf.where(tp < tp.shift(), 0).rolling(window=window).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf))
    data[f'MFI_{window}'] = mfi
    return data

def add_obv(data):
    """
    Adds On-Balance Volume (OBV) to the data.
    OBV uses volume flow to predict changes in stock price.
    """
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    return data

def add_vwap(data):
    """
    Adds Volume Weighted Average Price (VWAP) to the data.
    VWAP is a trading benchmark that gives the average price a security has traded at throughout the day, based on both volume and price.
    """
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    return data

def add_tsi(data, r, s):
    """
    Adds True Strength Index (TSI) to the data.
    TSI is a momentum oscillator that ranges between values of -100 and +100.
    """
    m = data['Close'].diff()
    m1 = m.ewm(span=r, adjust=False).mean()
    m2 = m1.ewm(span=s, adjust=False).mean()
    abs_m = abs(m)
    abs_m1 = abs_m.ewm(span=r, adjust=False).mean()
    abs_m2 = abs_m1.ewm(span=s, adjust=False).mean()
    data['TSI'] = 100 * (m2 / abs_m2)
    return data

def add_ultimate_oscillator(data, s, m, l):
    """
    Adds Ultimate Oscillator to the data.
    Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different timeframes.
    """
    bp = data['Close'] - np.minimum(data['Low'], data['Close'].shift())
    tr = np.maximum(data['High'] - data['Low'], np.maximum(abs(data['High'] - data['Close'].shift()), abs(data['Low'] - data['Close'].shift())))
    avg7 = bp.rolling(window=s).sum() / tr.rolling(window=s).sum()
    avg14 = bp.rolling(window=m).sum() / tr.rolling(window=m).sum()
    avg28 = bp.rolling(window=l).sum() / tr.rolling(window=l).sum()
    data['Ultimate_Oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
    return data

def add_keltner_channel(data, window, atr_window):
    """
    Adds Keltner Channel to the data.
    Keltner Channel is a volatility-based envelope set above and below an exponential moving average.
    """
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    atr = add_atr(data, atr_window)[f'ATR_{atr_window}']
    data[f'Keltner_Upper_{window}'] = ema + 2 * atr
    data[f'Keltner_Lower_{window}'] = ema - 2 * atr
    return data

def add_donchian_channel(data, window):
    """
    Adds Donchian Channel to the data.
    Donchian Channel is a moving average indicator developed by Richard Donchian.
    """
    data[f'Donchian_High_{window}'] = data['High'].rolling(window=window).max()