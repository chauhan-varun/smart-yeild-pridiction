"""
Standalone prediction script for testing the model locally
"""
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_model(model_path):
    """Load the trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

def preprocess_input(input_data, sample_data_path):
    """
    Preprocess input data for prediction.
    
    Args:
        input_data (dict or pandas.DataFrame): Input data with feature values
        sample_data_path (str): Path to sample training data for encoder fitting
        
    Returns:
        pandas.DataFrame: Processed input ready for prediction
    """
    # Convert input to DataFrame if it's a dict
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Load sample data for fitting encoder and scaler
    sample_data = pd.read_csv(sample_data_path, nrows=1000)
    # Clean column names
    sample_data.columns = [col.strip() for col in sample_data.columns]
    
    # Separate numeric and categorical features
    categorical_features = ['State', 'District', 'Crop', 'Season']
    numeric_features = ['Area', 'Production', 'Crop_Year']
    
    # Initialize encoder and scaler
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()
    
    # Fit encoder and scaler on sample data
    encoder.fit(sample_data[categorical_features])
    scaler.fit(sample_data[numeric_features])
    
    # Encode categorical features
    encoded_cats = encoder.transform(input_df[categorical_features])
    cat_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_cats, columns=cat_feature_names, index=input_df.index)
    
    # Scale numeric features
    scaled_nums = scaler.transform(input_df[numeric_features])
    input_df[numeric_features] = scaled_nums
    
    # Combine processed features
    processed_df = pd.concat([input_df[numeric_features], encoded_df], axis=1)
    
    return processed_df

def predict_yield(input_data, model_path='models/crop_yield_xgboost_model.joblib', sample_data_path='crop_yield_train.csv'):
    """
    Make a yield prediction using the trained model.
    
    Args:
        input_data (dict or pandas.DataFrame): Input data with feature values
        model_path (str): Path to the saved model
        sample_data_path (str): Path to sample training data for encoder fitting
        
    Returns:
        float or numpy.ndarray: Predicted yield(s)
    """
    # Load model
    model = load_model(model_path)
    
    # Preprocess input
    processed_input = preprocess_input(input_data, sample_data_path)
    
    # Make prediction
    prediction = model.predict(processed_input)
    
    # Return single value if input was a dict, otherwise return array
    if isinstance(input_data, dict):
        return float(prediction[0])
    else:
        return prediction

def main():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(description='Predict crop yield using trained model')
    parser.add_argument('--model', type=str, default='models/crop_yield_xgboost_model.joblib',
                        help='Path to the saved model file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to CSV file with input data (optional)')
    parser.add_argument('--state', type=str, default=None, help='State name')
    parser.add_argument('--district', type=str, default=None, help='District name')
    parser.add_argument('--crop', type=str, default=None, help='Crop name')
    parser.add_argument('--year', type=int, default=None, help='Crop year')
    parser.add_argument('--season', type=str, default=None, help='Season')
    parser.add_argument('--area', type=float, default=None, help='Area in hectares')
    parser.add_argument('--production', type=float, default=None, help='Production in tonnes')
    parser.add_argument('--sample-data', type=str, default='crop_yield_train.csv',
                        help='Path to sample training data for encoder fitting')
    
    args = parser.parse_args()
    
    # Check if input file is provided
    if args.input:
        # Load input data from CSV
        input_data = pd.read_csv(args.input)
        # Clean column names
        input_data.columns = [col.strip() for col in input_data.columns]
        
        # Make batch predictions
        predictions = predict_yield(input_data, args.model, args.sample_data)
        
        # Print results
        print("\nBatch Predictions:")
        for i, (_, row) in enumerate(input_data.iterrows()):
            print(f"Row {i+1}: {row['State']}, {row['District']}, {row['Crop']} - Predicted Yield: {predictions[i]:.2f}")
        
        print(f"\nAverage Predicted Yield: {np.mean(predictions):.2f}")
    
    # Check if individual parameters are provided
    elif all([args.state, args.district, args.crop, args.year, args.season, args.area is not None, args.production is not None]):
        # Create input data dictionary
        input_data = {
            'State': args.state,
            'District': args.district,
            'Crop': args.crop,
            'Crop_Year': args.year,
            'Season': args.season,
            'Area': args.area,
            'Production': args.production
        }
        
        # Make prediction
        prediction = predict_yield(input_data, args.model, args.sample_data)
        
        # Print result
        print("\nPrediction Result:")
        print(f"State: {args.state}")
        print(f"District: {args.district}")
        print(f"Crop: {args.crop}")
        print(f"Year: {args.year}")
        print(f"Season: {args.season}")
        print(f"Area: {args.area} hectares")
        print(f"Production: {args.production} tonnes")
        print(f"Predicted Yield: {prediction:.2f} tonnes per hectare")
    
    else:
        parser.print_help()
        print("\nError: You must either provide an input CSV file or all individual parameters.")

if __name__ == '__main__':
    main()
