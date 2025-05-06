# Deployment Guide for Smart Agricultural Yield Prediction

This guide covers how to test the model locally and deploy the web application to Render.

## Local Testing

### 1. Install Dependencies

First, install all the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Test the Model Locally

You can test the trained model using the `predict.py` script:

```bash
# Test with individual parameters
python predict.py --state "Punjab" --district "LUDHIANA" --crop "Rice" --year 2022 --season "Kharif" --area 100 --production 500

# Or test with a CSV file containing multiple records
python predict.py --input test_samples.csv
```

### 3. Run the Web Application Locally

To test the full web application on your local machine:

```bash
# Run the Flask application
python app.py
```

This will start the server at http://localhost:5000. You can access it in your web browser.

## Deploying to Render

### 1. Create a Render Account

If you don't already have one, sign up for a free Render account at https://render.com.

### 2. Connect your GitHub Repository

1. Push your code to a GitHub repository
2. In the Render dashboard, click "New Web Service"
3. Connect your GitHub account and select your repository

### 3. Configure the Web Service

Fill in the following configuration details:

- **Name**: smart-agricultural-yield-prediction (or your preferred name)
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Plan**: Free (or choose a paid plan for production use)

### 4. Advanced Settings (Optional)

You may want to configure these additional settings:

- **Environment Variables**: Add any environment variables if needed
- **Auto-Deploy**: Enable automatic deployments when you push to your repository

### 5. Deploy the Application

Click "Create Web Service" to start the deployment process. Render will build and deploy your application automatically.

## Testing Your Deployed Application

Once deployed, Render will provide you with a URL for your application (e.g., `https://smart-agricultural-yield-prediction.onrender.com`).

Visit this URL in your browser to access your application. You can use the web interface to make predictions or use the API endpoint for programmatic access.

### API Usage

You can use the prediction API from other applications:

```python
import requests
import json

# Replace with your actual Render URL
url = "https://smart-agricultural-yield-prediction.onrender.com/predict"

# Prepare data
data = {
    "State": "Punjab",
    "District": "LUDHIANA",
    "Crop": "Rice",
    "Crop_Year": 2022,
    "Season": "Kharif",
    "Area": 100,
    "Production": 500
}

# Make prediction request
response = requests.post(url, json=data)

# Print results
if response.status_code == 200:
    result = response.json()
    print(f"Predicted Yield: {result['predicted_yield']:.2f} tonnes per hectare")
else:
    print(f"Error: {response.text}")
```

## Health Monitoring

Your application includes a health check endpoint at `/health`. You can set up Render health checks to monitor your application:

- URL Path: `/health`
- Status: 200 OK

## Common Issues and Troubleshooting

- **Model Loading Error**: Ensure the model file is correctly included in your repository or uploaded to Render.
- **Memory Limits**: The Free tier on Render has memory limitations. If your model is large, consider optimizing it or upgrading to a paid plan.
- **Slow Initial Response**: Free tier services on Render spin down after inactivity. The first request after inactivity may be slow.

## Production Considerations

For production use, consider the following:

1. **Upgrade to a Paid Plan**: For better performance and uptime
2. **Custom Domain**: Configure a custom domain for your application
3. **SSL Certificate**: Render provides free SSL certificates
4. **Database**: Add a database for storing predictions and user data
5. **Authentication**: Implement user authentication for secure access
