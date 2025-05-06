"""
Smart Agricultural Yield Prediction System
Main entry point for the application
"""
import os
import time
import pandas as pd
import numpy as np
import dask.dataframe as dd
import warnings

# Handle potential import errors for optional modules
try:
    from dask.distributed import Client, LocalCluster
    DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:
    print("Warning: dask.distributed not available. Using single-threaded processing.")
    DASK_DISTRIBUTED_AVAILABLE = False

import argparse

# Import custom modules
from data_preprocessing import (
    load_data, 
    preprocess_data, 
    prepare_features_target,
    encode_categorical_features,
    scale_numeric_features
)
from visualization import create_visualizations
from modeling import (
    split_data, 
    train_model, 
    evaluate_model, 
    save_model,
    plot_feature_importance,
    plot_predictions_vs_actual
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Smart Agricultural Yield Prediction')
    parser.add_argument('--data', type=str, default='crop_yield_train.csv',
                        help='Path to the CSV data file')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest', 'gradient_boosting', 'linear_regression'],
                        help='Model type to use for prediction')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--sample', type=float, default=1.0,
                        help='Fraction of data to sample for training')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of Dask workers')
    return parser.parse_args()


def main():
    """Main function to run the crop yield prediction pipeline."""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    print("="*80)
    print("SMART AGRICULTURAL YIELD PREDICTION")
    print("="*80)
    
    # Set up Dask distributed computing if available
    client = None
    cluster = None
    
    if DASK_DISTRIBUTED_AVAILABLE:
        try:
            print(f"\nSetting up Dask with {args.workers} workers...")
            cluster = LocalCluster(n_workers=args.workers, threads_per_worker=2)
            client = Client(cluster)
            print(f"Dashboard link: {client.dashboard_link}")
        except Exception as e:
            print(f"Warning: Failed to set up Dask distributed computing: {e}")
            print("Falling back to single-threaded processing.")
            client = None
            cluster = None
    
    try:
        # Create directories for output
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Load data
        df = load_data(args.data)
        
        # Display basic dataset information
        print("\nDataset Information:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Size: {len(df)} rows")
        
        # Sample data if specified
        if args.sample < 1.0:
            print(f"\nSampling {args.sample*100:.1f}% of data...")
            df = df.sample(frac=args.sample)
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df_processed)
        
        # Prepare features and target
        X, y = prepare_features_target(df_processed, target_col='Yield')
        
        # Split data - this will convert to pandas
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=args.test_size
        )
        
        # No need to compute Dask DataFrames anymore since split_data already returns pandas DataFrames
        print("\nPreparing features for modeling...")
        
        # Encode categorical features
        categorical_features = ['State', 'District', 'Crop', 'Season']
        numeric_features = ['Area', 'Production', 'Crop_Year']
        
        # Handle encoding with pandas DataFrames
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        # For categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_data_train = encoder.fit_transform(X_train[categorical_features])
        categorical_data_test = encoder.transform(X_test[categorical_features])
        
        # Get feature names
        try:
            cat_feature_names = encoder.get_feature_names_out(categorical_features)
        except AttributeError:
            # For older scikit-learn versions
            cat_feature_names = [f"{col}_{val}" for i, col in enumerate(categorical_features)
                              for val in encoder.categories_[i]]
        
        # Convert to DataFrame
        categorical_df_train = pd.DataFrame(
            categorical_data_train, 
            columns=cat_feature_names,
            index=X_train.index
        )
        categorical_df_test = pd.DataFrame(
            categorical_data_test, 
            columns=cat_feature_names,
            index=X_test.index
        )
        
        # For numeric features
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # Combine numeric and encoded categorical features
        X_train_processed = pd.concat([X_train[numeric_features], categorical_df_train], axis=1)
        X_test_processed = pd.concat([X_test[numeric_features], categorical_df_test], axis=1)
        
        # Train model
        print(f"\nTraining {args.model} model...")
        model = train_model(X_train_processed, y_train, model_type=args.model)
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics, y_pred = evaluate_model(model, X_test_processed, y_test)
        
        # Plot feature importance
        plot_feature_importance(model, X_train_processed, args.model)
        
        # Plot predictions vs actual
        plot_predictions_vs_actual(y_test, y_pred, args.model)
        
        # Save model
        if not args.no_save:
            model_path = save_model(model, args.model)
        
        # Print execution time
        execution_time = time.time() - start_time
        print(f"\nExecution completed in {execution_time:.2f} seconds")
        
    finally:
        # Close the Dask client
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
            print("\nDask client and cluster closed")


if __name__ == "__main__":
    main()
