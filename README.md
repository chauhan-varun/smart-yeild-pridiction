# Smart Agricultural Yield Prediction System

A comprehensive web application for predicting agricultural crop yields using machine learning and data analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Model Details](#model-details)
- [MongoDB Integration](#mongodb-integration)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)

## Overview

The Smart Agricultural Yield Prediction System is designed to help farmers, agricultural researchers, and policymakers make data-driven decisions by predicting crop yields based on various parameters such as location, crop type, season, and cultivation area. The system uses advanced machine learning techniques to analyze historical data and provide accurate yield predictions, along with visual insights.

## Features

- **Accurate Yield Predictions**: Predict crop yields with XGBoost model achieving 91.78% accuracy (R² score)
- **Interactive Web Interface**: User-friendly web application for submitting prediction requests
- **MongoDB Integration**: Store and retrieve predictions and user feedback
- **Data Visualization**: Interactive charts displaying yield trends by crop, season, state, and year
- **Historical Analysis**: Track past predictions and analyze trends
- **User Feedback System**: Collect actual yield data to improve model accuracy
- **Future Predictions**: Project crop yields for future years (up to 10 years in advance)
- **Analytics Dashboard**: Comprehensive analytics on prediction data and user feedback
- **Responsive Design**: Mobile-friendly interface accessible across devices

## Technology Stack

### Backend
- **Python**: Core programming language
- **Flask**: Web application framework
- **Dask**: Parallel computing library for handling large datasets
- **XGBoost**: Gradient boosting algorithm for prediction model
- **Scikit-learn**: Machine learning library for preprocessing and evaluation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **PyMongo**: MongoDB integration

### Frontend
- **HTML5/CSS3**: Frontend structure and styling
- **JavaScript**: Client-side interactivity
- **Bootstrap**: Responsive UI framework
- **Font Awesome**: Icon library
- **Chart.js**: Interactive data visualization

### Database
- **MongoDB**: NoSQL database for storing predictions and feedback

### Dependencies
- **python-dotenv**: Environment variable management
- **joblib**: Model serialization
- **io/BytesIO**: In-memory file handling
- **base64**: Image encoding for web display
- **uuid**: Unique identifier generation

## Model Details

The system uses an **XGBoost** (Extreme Gradient Boosting) model for predicting crop yields:

- **Accuracy**: 91.78% (R² score)
- **Features**: 
  - State
  - District
  - Crop type
  - Season
  - Cultivation area
  - Production amount
  - Crop year
- **Preprocessing**: 
  - One-hot encoding for categorical features (State, District, Crop, Season)
  - Standard scaling for numerical features (Area, Production, Crop_Year)
- **Cross-validation**: 5-fold cross-validation for robustness
- **Hyperparameter Tuning**: Optimized model parameters for maximum accuracy

## MongoDB Integration

The system integrates with MongoDB to provide persistent storage:

- **Collections**:
  - **predictions**: Stores prediction records with timestamp, input parameters, and results
  - **feedback**: Captures user-reported actual yields to improve model accuracy
  - **users**: Optional user management for personalized experiences

- **Schema Example** (prediction document):
```json
{
  "prediction_id": "uuid-string",
  "predicted_yield": 23.5,
  "input_data": {
    "State": "Karnataka",
    "District": "BANGALORE",
    "Crop": "Rice",
    "Crop_Year": 2023,
    "Season": "Kharif",
    "Area": 150,
    "Production": 3500
  },
  "timestamp": "2025-04-16T12:00:00",
  "user_id": "user-uuid-string"
}
```

## Project Structure

```
crop_yield_train/
├── app.py                  # Main Flask application
├── train_model.py          # Model training script
├── data_preprocessing.py   # Data processing utilities
├── models/                 # Saved model files
│   └── crop_yield_model.pkl
├── static/                 # Static assets
│   ├── css/
│   │   └── custom.css
│   └── js/
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   └── analytics.html
├── visualizations/         # Generated plots
├── .env                    # Environment variables
└── requirements.txt        # Project dependencies
```

## Installation and Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/crop_yield_train.git
cd crop_yield_train
```

2. **Set up a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Create .env file**:
```
MONGODB_URL=your_mongodb_connection_string
SECRET_KEY=your_secret_key
```

5. **Run the application**:
```bash
python app.py
```

6. **Access the web interface**:
Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

### Making Predictions

1. Navigate to the home page
2. Fill in the prediction form with:
   - State
   - District
   - Crop type
   - Year (including future years)
   - Season
   - Area (hectares)
   - Production (tonnes)
3. Click "Predict Yield"
4. View the prediction results and visualization

### Providing Feedback

1. After receiving a prediction, click "Add Actual Yield"
2. Enter the actual yield value once harvested
3. Optionally add comments about the harvest
4. Submit to improve future predictions

### Viewing Analytics

1. Click "Analytics Dashboard" from the navigation menu
2. Explore trends, patterns, and insights from all predictions
3. Analyze prediction accuracy based on user feedback

## API Reference

The system provides a RESTful API for programmatic access:

### Prediction Endpoint

**POST** `/predict`

Request body:
```json
{
  "State": "Karnataka",
  "District": "BANGALORE",
  "Crop": "Rice",
  "Crop_Year": 2023,
  "Season": "Kharif",
  "Area": 150,
  "Production": 3500
}
```

Response:
```json
{
  "prediction_id": "uuid-string",
  "predicted_yield": 23.5
}
```

### Feedback Endpoint

**POST** `/feedback`

Request body:
```json
{
  "prediction_id": "uuid-string",
  "actual_yield": 24.2,
  "comments": "Good harvest with timely rainfall"
}
```

## Deployment

The system is designed for deployment on various platforms:

### Render Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add environment variables (MONGODB_URL, SECRET_KEY)
5. Deploy the application

## Future Enhancements

1. **Climate Data Integration**: Incorporate weather forecasts for improved prediction accuracy
2. **User Authentication**: Implement secure login system for personalized experiences
3. **Crop Recommendations**: Suggest optimal crops based on location and season
4. **Mobile Application**: Develop native mobile apps for Android and iOS
5. **Advanced Analytics**: Implement machine learning for anomaly detection and pattern recognition
6. **Offline Support**: Enable offline functionality for rural areas with limited connectivity
7. **Multilingual Support**: Add language options for broader accessibility
8. **Satellite Imagery**: Integrate satellite data for real-time crop monitoring
