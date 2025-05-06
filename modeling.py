"""
Modeling functions for agricultural yield prediction.
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        X (dask.dataframe.DataFrame): Features
        y (dask.dataframe.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"Splitting data with test_size={test_size}...")
    
    # Convert to pandas for splitting
    X_pd = X.compute()
    y_pd = y.compute()
    
    # Use sklearn's train_test_split with pandas DataFrames
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y_pd, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='xgboost'):
    """
    Train a regression model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        model_type (str): Type of model to train
        
    Returns:
        object: Trained model
    """
    print(f"Training {model_type} model...")
    start_time = time.time()
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_type == 'linear_regression':
        model = LinearRegression(n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Model trained in {training_time:.2f} seconds")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model (object): Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create metrics dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    # Print metrics
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics, y_pred


def save_model(model, model_type):
    """
    Save model to disk.
    
    Args:
        model (object): Trained model
        model_type (str): Type of model
        
    Returns:
        str: Path to saved model
    """
    print("Saving model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Define model path
    model_path = f'models/crop_yield_{model_type}_model.joblib'
    
    # Save model
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")
    return model_path


def plot_feature_importance(model, X_train, model_type):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model (object): Trained model
        X_train (dask.dataframe.DataFrame): Training features
        model_type (str): Type of model
    """
    print("Plotting feature importance...")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    if model_type in ['xgboost', 'random_forest', 'gradient_boosting']:
        # Get feature importance
        if model_type == 'xgboost':
            importance = model.feature_importances_
        else:
            importance = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Get feature names
        feature_names = X_train.columns
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance ({model_type})')
        plt.bar(range(len(indices)), importance[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'visualizations/feature_importance_{model_type}.png')
        plt.close()
        
        print("Feature importance plot saved")
    else:
        print(f"Feature importance not available for {model_type}")


def plot_predictions_vs_actual(y_test, y_pred, model_type):
    """
    Plot predicted vs actual values.
    
    Args:
        y_test (pandas.Series): Actual values
        y_pred (numpy.ndarray): Predicted values
        model_type (str): Type of model
    """
    print("Plotting predictions vs actual values...")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted Yield ({model_type})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'visualizations/actual_vs_predicted_{model_type}.png')
    plt.close()
    
    print("Actual vs predicted plot saved")
