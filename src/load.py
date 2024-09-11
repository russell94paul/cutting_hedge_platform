# src/load.py
import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)