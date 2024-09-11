import sys
import os

# Add the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import extract as extract
import preprocess as preprocess
import feature_engineering as fe
import feature_scaling as fs
import feature_importance as fi
import model_training as mt
import model_evaluation as me
import model_predictions as mp
import backtest as bt



# Data Download Config
ticker = 'AMZN' 
start_date = '2020-01-01'
end_date = '2021-01-01'
file_path = f'data/raw/{ticker}_ohlcv.parquet'


# Extract Data from API and save in parquet file
ohlcv = extract.extract_ohlcv_data(ticker,  start_date, end_date, file_path)

# Preprocess raw data for feature engineering 
ohlcv = preprocess.preprocess_ohlcv_data(ohlcv)

# Add features - trading indicators, lagged features, alternative calculations
ohlcv = fe.add_technical_indicators(ohlcv)

# Feature scaling
# TO-DO : modify function to allow different scaling algorithms to be applied
ohlcv_scaled = fs.scale_features(ohlcv)

# Prepare data for training
X, y = mt.prepare_data(ohlcv_scaled)

# Model Training
model, X_train, X_test, y_train, y_test = mt.train_random_forest(X, y)

# Model Evaluation
metrics, y_pred = me.evaluate_model(model, X_test, y_test)

# Display Metrics
me.display_metrics(metrics)

# Feature Importance
feature_importances = fi.calculate_feature_importance(model, X)
print(feature_importances.sort_values(ascending=False))
fi.plot_feature_importance(feature_importances, title='Feature Importance', save_path='src/features/data/feature_importance.png')

# Permutation Feature Importance
perm_importances = fi.calculate_permutation_importance(model, X, y)
print(perm_importances.sort_values(ascending=False))
fi.plot_feature_importance(perm_importances, title='Permutation Feature Importance', save_path='src/features/data/permutation_feature_importance.png')

# Make Predictions
ohlcv = mp.make_predictions(model, X, ohlcv)

# Backtesting - model prediction simulation
ohlcv = bt.backtest_strategy(ohlcv)
print(ohlcv[['Predicted Return', 'Strategy Return', 'Cumulative Return']])

# Convert Predicted Returns to Signals - for vbt
ohlcv = bt.convert_to_signals(ohlcv)

# VectorBT Backtesting - Trade Simulation
portfolio = bt.vectorbt_backtest(ohlcv)
print(portfolio.stats())

# TO-DO: Create dashboard to display stats (using streamlit - if possible)

# ADDITIONAL TO-DO:

## 1. Confirm How ML model can be converted into trading signals (buys/sells, entries/exits)
### 1.1. may need a strategy.py module

## 2. Add analysis notebooks
### 2.2. Indicator Feasibility Analysis
### 2.3. Monte-Carlo Simulation (Strategy Robustness Testing)

## 3. Strategy Optimization
### 3.1. Hypterparameter Optimzation
### 3.2. Walkforward Optimization

## 4. Risk management module - see CodeTrading YouTube Channel