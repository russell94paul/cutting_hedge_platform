import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(ohlcv, target_column='Daily Return'):
    """
    Scale features in the dataframe using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing features and target.
    target_column (str): Name of the target column to exclude from scaling.
    
    Returns:
    pd.DataFrame: Dataframe with scaled features.
    """
    scaler = StandardScaler()
    
    # Separate features and target
    features = ohlcv.drop(columns=[target_column])
    target = ohlcv[target_column]
    
    # Scale features
    scaled_features = scaler.fit_transform(features)
    
    # Convert scaled features back to DataFrame
    scaled_ohlcv = pd.DataFrame(scaled_features, columns=features.columns, index=ohlcv.index)
    scaled_ohlcv[target_column] = target
    
    return scaled_ohlcv
    