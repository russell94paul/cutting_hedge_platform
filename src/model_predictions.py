import pandas as pd

def make_predictions(model, X, df, prediction_column='Predicted Return'):
    """
    Make predictions using the trained model and add them to the dataframe.
    
    Parameters:
    model: Trained model.
    X (pd.DataFrame): Features.
    df (pd.DataFrame): Dataframe to add predictions to.
    prediction_column (str): Column name for predicted returns.
    
    Returns:
    pd.DataFrame: Dataframe with added predictions.
    """
    df[prediction_column] = model.predict(X)
    return df