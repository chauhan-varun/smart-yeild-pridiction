"""
Flask application for Smart Agricultural Yield Prediction
Provides both API endpoints and a web interface
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for, flash
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import datetime
import uuid
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_url_path='/static', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# MongoDB setup
MONGODB_URL = os.environ.get('MONGODB_URL')
if not MONGODB_URL:
    app.logger.warning("No MongoDB URL found. Using in-memory predictions only.")
    mongodb_client = None
    db = None
else:
    try:
        mongodb_client = MongoClient(MONGODB_URL)
        # Get database (or create if not exists)
        db = mongodb_client.crop_yield
        app.logger.info("Connected to MongoDB successfully")
    except Exception as e:
        app.logger.error(f"MongoDB connection error: {e}")
        mongodb_client = None
        db = None

def load_model():
    """
    Load the pre-trained model.
    
    Returns:
        object: Loaded model
    """
    import joblib
    
    # Ensure the model file exists
    if not os.path.exists('models/crop_yield_model.pkl'):
        # If model doesn't exist, train a new one with a subset of data
        app.logger.warning("Model not found, training a new model with sample data...")
        from train_model import train_model
        train_model(sample=True)  # Train with a sample of data for quick startup
    
    try:
        model_data = joblib.load('models/crop_yield_model.pkl')
        model = model_data['model']
        feature_names = model_data.get('feature_names', None)
        app.logger.info(f"Model loaded successfully with feature names: {feature_names}")
        return model, feature_names
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        raise Exception(f"Failed to load model: {e}")

# Load the model at startup
model, expected_feature_names = load_model()

# Cache encoders and scalers
encoder = None
scaler = None

def create_preprocessors():
    """
    Create or load preprocessors for categorical and numeric features.
    """
    # Define the encoder and scaler
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    scaler = StandardScaler()
    
    # Load a sample of training data to fit the preprocessors if needed
    sample_data = pd.read_csv('crop_yield_train.csv', nrows=1000)
    
    # Clean column names
    sample_data.columns = [col.strip() for col in sample_data.columns]
    
    # Fit the encoder on all possible categorical values from training data
    categorical_features = ['State', 'District', 'Crop', 'Season']
    encoder.fit(sample_data[categorical_features])
    
    # Fit the scaler on numeric features
    numeric_features = ['Area', 'Production', 'Crop_Year']
    scaler.fit(sample_data[numeric_features])
    
    return encoder, scaler

def preprocess_input(input_data):
    """
    Preprocess input data for prediction.
    
    Args:
        input_data (dict): Input data with feature values
        
    Returns:
        pandas.DataFrame: Processed input ready for prediction
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns are present
    required_columns = ['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area', 'Production']
    for col in required_columns:
        if col not in input_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Separate numeric and categorical features
    categorical_features = ['State', 'District', 'Crop', 'Season']
    numeric_features = ['Area', 'Production', 'Crop_Year']
    
    # Get preprocessors
    encoder, scaler = create_preprocessors()
    
    # Encode categorical features
    try:
        encoded_cats = encoder.transform(input_df[categorical_features])
        
        # Generate feature names in the expected format
        cat_feature_names = []
        for i, feature in enumerate(categorical_features):
            for category in encoder.categories_[i]:
                cat_feature_names.append(f"{feature}_{category}")
        
        encoded_df = pd.DataFrame(encoded_cats, columns=cat_feature_names, index=input_df.index)
    except Exception as e:
        app.logger.error(f"Error encoding categorical features: {e}")
        raise ValueError(f"Error encoding categorical features: {e}")
    
    # Scale numeric features
    try:
        scaled_nums = scaler.transform(input_df[numeric_features])
        scaled_df = pd.DataFrame(scaled_nums, columns=numeric_features, index=input_df.index)
    except Exception as e:
        app.logger.error(f"Error scaling numeric features: {e}")
        raise ValueError(f"Error scaling numeric features: {e}")
    
    # Combine processed features - put numeric features first then categorical
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)
    
    # Log the feature names for debugging
    app.logger.info(f"Processed feature names: {processed_df.columns.tolist()}")
    
    return processed_df

def make_prediction(input_data, user_id=None):
    """
    Process input, make prediction, and save to MongoDB.
    
    Args:
        input_data (dict): Input features
        user_id (str, optional): User identifier
        
    Returns:
        tuple: (prediction, prediction_id, plot_data)
    """
    # Preprocess input
    processed_input = preprocess_input(input_data)
    
    # Get expected feature names for the model
    global expected_feature_names
    
    # Reorder columns to match the expected feature names if they exist
    if expected_feature_names is not None:
        # Check which expected features are present in our processed input
        available_features = [f for f in expected_feature_names if f in processed_input.columns]
        missing_features = [f for f in expected_feature_names if f not in processed_input.columns]
        
        if missing_features:
            app.logger.warning(f"Missing expected features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                processed_input[feature] = 0
        
        # Reorder to match expected features
        processed_input = processed_input[expected_feature_names]
    
    # Make prediction
    prediction = model.predict(processed_input)[0]
    
    # Generate prediction ID
    prediction_id = str(uuid.uuid4())
    
    # Create prediction record for MongoDB
    prediction_record = {
        'prediction_id': prediction_id,
        'predicted_yield': float(prediction),
        'input_data': input_data,
        'timestamp': datetime.datetime.utcnow(),
        'user_id': user_id
    }
    
    # Save to MongoDB
    if mongodb_client:
        try:
            predictions_collection = db.predictions
            predictions_collection.insert_one(prediction_record)
            app.logger.info(f"Prediction saved to MongoDB with ID: {prediction_id}")
        except Exception as e:
            app.logger.error(f"Failed to save prediction to MongoDB: {e}")
    
    # Create plot data
    plot_data = generate_yield_distribution_plot(input_data['Crop'])
    
    return prediction, prediction_id, plot_data

def generate_yield_distribution_plot(crop_name):
    """
    Generate a distribution plot for a specific crop's yield.
    
    Args:
        crop_name (str): Name of the crop
        
    Returns:
        str: Base64 encoded image data
    """
    try:
        # Load sample data
        sample_data = pd.read_csv('crop_yield_train.csv', nrows=5000)
        sample_data.columns = [col.strip() for col in sample_data.columns]
        
        # Filter for the specific crop
        crop_data = sample_data[sample_data['Crop'] == crop_name]
        
        if len(crop_data) == 0:
            app.logger.warning(f"No data available for crop: {crop_name}")
            return None
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.histplot(crop_data['Yield'], kde=True)
        plt.title(f'Yield Distribution for {crop_name}')
        plt.xlabel('Yield (tonnes per hectare)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 for embedding in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return plot_data
    except Exception as e:
        app.logger.error(f"Error generating plot: {e}")
        return None

def get_user_predictions(user_id, limit=10):
    """
    Get previous predictions for a user.
    
    Args:
        user_id (str): User ID
        limit (int): Maximum number of predictions to return
        
    Returns:
        list: List of prediction records
    """
    if not user_id:
        return []
    
    # Query MongoDB for user's predictions
    cursor = db.predictions.find(
        {'user_id': user_id}
    ).sort('timestamp', -1).limit(limit)
    
    # Convert MongoDB documents to list of dictionaries
    predictions = []
    for doc in cursor:
        doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
        predictions.append(doc)
    
    return predictions

def save_feedback(prediction_id, actual_yield, comments=None, user_id=None):
    """
    Save user feedback for a prediction.
    
    Args:
        prediction_id (str): ID of the prediction
        actual_yield (float): Actual yield reported by the user
        comments (str, optional): User comments
        user_id (str, optional): ID of the user providing feedback
        
    Returns:
        str: ID of the feedback record
    """
    # Create feedback record
    feedback = {
        'prediction_id': prediction_id,
        'actual_yield': float(actual_yield),
        'predicted_yield': None,  # Will be filled in
        'absolute_error': None,   # Will be calculated
        'comments': comments,
        'user_id': user_id,
        'timestamp': datetime.datetime.utcnow()
    }
    
    # Get the original prediction
    prediction = db.predictions.find_one({'prediction_id': prediction_id})
    
    if prediction:
        feedback['predicted_yield'] = prediction.get('predicted_yield')
        # Calculate absolute error
        feedback['absolute_error'] = abs(float(actual_yield) - float(prediction.get('predicted_yield', 0)))
    
    # Store feedback in MongoDB
    result = db.feedback.insert_one(feedback)
    
    return str(result.inserted_id)

# A few helper functions
def get_unique_values(column_name):
    """
    Get unique values from a column in the dataset.
    Used for populating dropdown menus.
    
    Args:
        column_name (str): Name of the column
        
    Returns:
        list: Sorted list of unique values
    """
    try:
        # Load a sample of the dataset
        sample_data = pd.read_csv('crop_yield_train.csv', nrows=5000)
        
        # Clean column names
        sample_data.columns = [col.strip() for col in sample_data.columns]
        
        # Get unique values
        if column_name in sample_data.columns:
            return sorted(sample_data[column_name].unique().tolist())
        else:
            app.logger.error(f"Column '{column_name}' not found in dataset")
            return []
    except Exception as e:
        app.logger.error(f"Error getting unique values for {column_name}: {e}")
        return []

@app.route('/')
def home():
    """Render the home page."""
    # Get unique values for dropdown menus
    states = get_unique_values('State')
    districts = get_unique_values('District')
    crops = get_unique_values('Crop')
    seasons = get_unique_values('Season')
    
    # Get min and max years from the dataset
    min_year = 2000  # Set a reasonable minimum year
    current_year = datetime.datetime.now().year
    max_year = current_year + 10  # Allow predictions up to 10 years in the future
    
    # Get user's previous predictions if logged in
    user_id = session.get('user_id')
    previous_predictions = get_user_predictions(user_id, limit=5) if user_id else []
    
    return render_template('index.html', 
                          states=states,
                          districts=districts,
                          crops=crops, 
                          seasons=seasons,
                          min_year=min_year,
                          max_year=max_year,
                          previous_predictions=previous_predictions,
                          user_id=user_id)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        # Get input data from request
        input_data = request.get_json()
        
        # Get user ID from session if available
        user_id = session.get('user_id')
        
        # Make prediction and store in MongoDB
        result = make_prediction(input_data, user_id)
        
        # Return prediction
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form-based prediction requests."""
    try:
        # Extract input data from form
        input_data = {
            'State': request.form['state'],
            'District': request.form['district'],
            'Crop': request.form['crop'],
            'Crop_Year': int(request.form['crop_year']),
            'Season': request.form['season'],
            'Area': float(request.form['area']),
            'Production': float(request.form['production'])
        }
        
        # Get user ID from session or generate a temporary one
        user_id = session.get('user_id', str(uuid.uuid4()))
        session['user_id'] = user_id
        
        # Make prediction
        prediction, prediction_id, plot_data = make_prediction(input_data, user_id)
        
        # Render the result template
        return render_template(
            'result.html', 
            prediction=prediction, 
            input_data=input_data, 
            plot_data=plot_data,
            prediction_id=prediction_id,
            user_id=user_id
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                              error=f"Prediction error: {str(e)}",
                              states=get_unique_values('State'),
                              districts=get_unique_values('District'),
                              crops=get_unique_values('Crop'),
                              seasons=get_unique_values('Season'),
                              min_year=2000,
                              max_year=2023)

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle user feedback submission."""
    try:
        prediction_id = request.form.get('prediction_id')
        actual_yield = float(request.form.get('actual_yield'))
        comments = request.form.get('comments')
        user_id = session.get('user_id')
        
        if not prediction_id:
            raise ValueError("Missing prediction ID")
        
        # Save feedback to MongoDB
        feedback_id = save_feedback(prediction_id, actual_yield, comments, user_id)
        
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('history'))
    
    except Exception as e:
        flash(f'Error submitting feedback: {str(e)}', 'danger')
        return redirect(url_for('history'))

@app.route('/history')
def history():
    """Show prediction history for the user."""
    user_id = session.get('user_id')
    
    if not user_id:
        flash('Please login to view your prediction history', 'warning')
        return redirect(url_for('home'))
    
    # Get user's predictions
    predictions = get_user_predictions(user_id, limit=50)
    
    return render_template('history.html', predictions=predictions)

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization images."""
    return send_from_directory('visualizations', filename)

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    # Check MongoDB connection
    try:
        # Ping MongoDB
        mongodb_client.admin.command('ping')
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'
    
    return jsonify({
        'status': 'healthy', 
        'message': 'Service is up and running',
        'database': db_status
    })

@app.route('/analytics')
def analytics():
    """Display analytics based on stored predictions."""
    try:
        # Get basic statistics
        total_predictions = db.predictions.count_documents({})
        
        # Get average predicted yield by crop
        pipeline = [
            {"$group": {
                "_id": "$input_data.Crop",
                "avg_yield": {"$avg": "$predicted_yield"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        crop_stats = list(db.predictions.aggregate(pipeline))
        
        # Get prediction counts by date
        pipeline = [
            {"$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "predicted_yield": 1
            }},
            {"$group": {
                "_id": "$date",
                "count": {"$sum": 1},
                "avg_yield": {"$avg": "$predicted_yield"}
            }},
            {"$sort": {"_id": 1}},
            {"$limit": 30}
        ]
        date_stats = list(db.predictions.aggregate(pipeline))
        
        # Get prediction accuracy based on feedback
        pipeline = [
            {"$lookup": {
                "from": "predictions",
                "localField": "prediction_id",
                "foreignField": "prediction_id",
                "as": "prediction"
            }},
            {"$unwind": "$prediction"},
            {"$project": {
                "crop": "$prediction.input_data.Crop",
                "predicted_yield": "$predicted_yield",
                "actual_yield": "$actual_yield",
                "absolute_error": "$absolute_error"
            }},
            {"$group": {
                "_id": "$crop",
                "avg_error": {"$avg": "$absolute_error"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        accuracy_stats = list(db.feedback.aggregate(pipeline))
        
        return render_template('analytics.html', 
                              total_predictions=total_predictions,
                              crop_stats=crop_stats,
                              date_stats=date_stats,
                              accuracy_stats=accuracy_stats)
    
    except Exception as e:
        return render_template('analytics.html', error=str(e))

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask application in debug mode for development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
