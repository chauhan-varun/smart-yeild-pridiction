"""
Data preprocessing functions for agricultural yield prediction.
"""
import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """
    Load data from CSV file using Dask DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dask.dataframe.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")
    # Set dtype for better memory usage and performance
    df = dd.read_csv(file_path)
    print(f"Data loaded with {len(df.columns)} columns")
    return df


def preprocess_data(df):
    """
    Preprocess the data for modeling.
    
    Args:
        df (dask.dataframe.DataFrame): Input data
        
    Returns:
        dask.dataframe.DataFrame: Preprocessed data
    """
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Clean column names by stripping whitespace
    df_processed = df_processed.rename(columns=lambda x: x.strip())
    
    # Strip whitespace from string columns
    for col in ['State', 'District', 'Crop', 'Season']:
        df_processed[col] = df_processed[col].str.strip()
    
    # Handle missing values
    numeric_cols = ['Area', 'Production', 'Yield']
    for col in numeric_cols:
        # Fill missing values with column median
        median = df_processed[col].dropna().compute().median()
        df_processed[col] = df_processed[col].fillna(median)
    
    print("Preprocessing completed")
    return df_processed


def prepare_features_target(df, target_col='Yield'):
    """
    Prepare features and target variable for modeling.
    
    Args:
        df (dask.dataframe.DataFrame): Input data
        target_col (str): Name of the target column
        
    Returns:
        tuple: X (features), y (target)
    """
    print("Preparing features and target...")
    
    # Select features
    numeric_features = ['Area', 'Production', 'Crop_Year']
    categorical_features = ['State', 'District', 'Crop', 'Season']
    
    # Create feature set
    X = df[numeric_features + categorical_features]
    y = df[target_col]
    
    print(f"Features: {len(X.columns)}, Target: {target_col}")
    return X, y


def encode_categorical_features(X_train, X_test, cat_features):
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        X_train (dask.dataframe.DataFrame): Training features
        X_test (dask.dataframe.DataFrame): Test features
        cat_features (list): List of categorical feature names
        
    Returns:
        tuple: Encoded X_train and X_test
    """
    print("Encoding categorical features...")
    
    # Convert to pandas for encoding
    X_train_pd = X_train.compute()
    X_test_pd = X_test.compute()
    
    # Get numeric features
    numeric_features = [col for col in X_train.columns if col not in cat_features]
    
    # Initialize encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform categorical features
    encoded_train = encoder.fit_transform(X_train_pd[cat_features])
    encoded_test = encoder.transform(X_test_pd[cat_features])
    
    # Get feature names
    try:
        feature_names = encoder.get_feature_names_out(cat_features)
    except AttributeError:
        # For older scikit-learn versions
        feature_names = [f"{col}_{val}" for i, col in enumerate(cat_features) 
                        for val in encoder.categories_[i]]
    
    # Convert to pandas DataFrames
    encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index=X_train_pd.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index=X_test_pd.index)
    
    # Combine with numeric features
    X_train_encoded = pd.concat([X_train_pd[numeric_features], encoded_train_df], axis=1)
    X_test_encoded = pd.concat([X_test_pd[numeric_features], encoded_test_df], axis=1)
    
    print(f"Encoded features: {X_train_encoded.shape[1]}")
    return X_train_encoded, X_test_encoded


def scale_numeric_features(X_train, X_test, numeric_features):
    """
    Scale numeric features using StandardScaler.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Test features
        numeric_features (list): List of numeric feature names
        
    Returns:
        tuple: Scaled X_train and X_test
    """
    print("Scaling numeric features...")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform numeric features
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print("Scaling completed")
    return X_train, X_test
