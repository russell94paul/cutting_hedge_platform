from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def prepare_data(df, target_column='Daily Return'):
    """
    Prepare data for training by defining features and target.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing features and target.
    target_column (str): Name of the target column.
    
    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_random_forest(X, y, test_size=0.2, random_state=42, n_estimators=100):
    """
    Train a Random Forest model.
    
    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed.
    n_estimators (int): Number of trees in the forest.
    
    Returns:
    model: Trained Random Forest model.
    X_train, X_test, y_train, y_test: Train-test split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test