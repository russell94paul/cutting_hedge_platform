from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.metrics._regression import root_mean_squared_error

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using various metrics.
    
    Parameters:
    model: Trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    
    Returns:
    metrics (dict): Dictionary containing various evaluation metrics.
    y_pred (pd.Series): Predicted values.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)
    
    metrics = {
        'Mean Squared Error (MSE)': mse,
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'R-squared (R²)': r2,
        'Mean Absolute Percentage Error (MAPE)': mape,
        'Explained Variance Score': explained_variance
    }
    
    return metrics, y_pred

def display_metrics(metrics):
    """
    Display the evaluation metrics and their explanations.
    
    Parameters:
    metrics (dict): Dictionary containing various evaluation metrics.
    """
    for metric, value in metrics.items():
        print(f'{metric}: {value}')
    
    # Explanation of each metric
    print("\nExplanation of each metric:")
    print("1. Mean Squared Error (MSE): Measures the average squared difference between the actual and predicted values. Lower values indicate better model performance.")
    print("2. Mean Absolute Error (MAE): Measures the average absolute difference between the actual and predicted values. Lower values indicate better model performance.")
    print("3. Root Mean Squared Error (RMSE): The square root of the average squared differences between the actual and predicted values. Lower values indicate better model performance.")
    print("4. R-squared (R²): Represents the proportion of the variance for the dependent variable that's explained by the independent variables in the model. Values closer to 1 indicate better model performance.")
    print("5. Mean Absolute Percentage Error (MAPE): Measures the average absolute percentage difference between the actual and predicted values. Lower values indicate better model performance.")
    print("6. Explained Variance Score: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables. Higher values indicate better model performance.")