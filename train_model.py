"""
Train a machine learning model for crop yield prediction.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def train_model(sample=False):
    """
    Train an XGBoost model for crop yield prediction.
    
    Args:
        sample (bool): If True, use a small sample of data for quick training
    
    Returns:
        None: Saves model and related data to disk
    """
    print("Loading and preprocessing data...")
    
    # Load data
    if sample:
        # Use a small sample for quick training
        data = pd.read_csv('crop_yield_train.csv', nrows=10000)
    else:
        # Use full dataset for production model
        data = pd.read_csv('crop_yield_train.csv')
    
    # Clean column names
    data.columns = [col.strip() for col in data.columns]
    
    # Define features and target
    X = data.drop('Yield', axis=1)
    y = data['Yield']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess data
    categorical_features = ['State', 'District', 'Crop', 'Season']
    numeric_features = ['Area', 'Production', 'Crop_Year']
    
    # Initialize transformers
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    scaler = StandardScaler()
    
    # Fit transformers
    X_train_cat = encoder.fit_transform(X_train[categorical_features])
    X_train_num = scaler.fit_transform(X_train[numeric_features])
    
    # Generate feature names
    cat_feature_names = []
    for i, feature in enumerate(categorical_features):
        for category in encoder.categories_[i]:
            cat_feature_names.append(f"{feature}_{category}")
    
    # Transform test data
    X_test_cat = encoder.transform(X_test[categorical_features])
    X_test_num = scaler.transform(X_test[numeric_features])
    
    # Combine features
    X_train_processed = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_test_processed = np.concatenate([X_test_num, X_test_cat], axis=1)
    
    # Create feature names list
    feature_names = numeric_features + cat_feature_names
    
    print(f"Training with {X_train_processed.shape[1]} features...")
    
    # Train XGBoost model
    params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror',
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_processed, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and related data
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'encoder': encoder,
        'scaler': scaler,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'metrics': {
            'rmse': rmse,
            'r2': r2
        }
    }
    
    joblib.dump(model_data, 'models/crop_yield_model.pkl')
    print("Model saved to models/crop_yield_model.pkl")

if __name__ == "__main__":
    train_model()
