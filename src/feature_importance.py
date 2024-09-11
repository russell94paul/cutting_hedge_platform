import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def calculate_feature_importance(model, X):
    """
    Calculate feature importance using the model's built-in feature importance.
    
    Parameters:
    model: Trained model.
    X (pd.DataFrame): Features.
    
    Returns:
    pd.Series: Feature importances.
    """
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances

def calculate_permutation_importance(model, X, y, n_repeats=30, random_state=42):
    """
    Calculate permutation feature importance.
    
    Parameters:
    model: Trained model.
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    n_repeats (int): Number of times to permute a feature.
    random_state (int): Random seed.
    
    Returns:
    pd.Series: Permutation feature importances.
    """
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    perm_importances = pd.Series(result.importances_mean, index=X.columns)
    return perm_importances

def plot_feature_importance(feature_importances, title='Feature Importance', save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    feature_importances (pd.Series): Feature importances.
    title (str): Title of the plot.
    save_path (str): Path to save the plot image. If None, the plot will be displayed.
    
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()