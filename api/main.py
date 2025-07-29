import uvicorn
from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import mlflow
import mlflow.keras
import pickle
import os

# --- FastAPI App and MLflow Setup ---

# Metadata for API documentation
tags_metadata = [
    {
        "name": "Amazon Stock Prediction",
        "description": "Predicting the next day's closing price for AMZN stock."
    }
]

app = FastAPI(
    title="Amazon Stock Prediction API",
    description="An API to predict the next day's closing price of Amazon (AMZN) stock using an LSTM model.",
    version="1.0",
    openapi_tags=tags_metadata
)

# Initialize MLflow
# Make sure the MLflow tracking server is running on this URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("amazon_stock_predictor")

# --- Load Model and Scaler ---

# Load the pre-trained Keras model
# Ensure 'best_model.h5' is in the same directory
try:
    model_path = "best_model.h5"
    model = load_model(model_path, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the scaler object from the pickle file
# Ensure 'scaler.pkl' is in the same directory
try:
    scaler_path = "scaler.pkl"
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# --- API Endpoints ---

@app.get("/")
def root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Amazon Stock Prediction API"}

@app.get("/predict", tags=["Amazon Stock Prediction"])
async def predict():
    """
    Fetches the last 60 days of AMZN stock data, preprocesses it,
    and returns the predicted closing price for the next day.
    """
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please check the files and restart the server."}

    # Start an MLflow run to log this prediction event
    with mlflow.start_run():
        
        # 1. Fetch historical data for the last 90 days to ensure we get 60 trading days
        ticker = yf.Ticker("AMZN")
        historical_data = ticker.history(period="90d", actions=False)
        
        # Ensure we have enough data
        if len(historical_data) < 60:
            return {"error": f"Not enough historical data available. Found only {len(historical_data)} days."}

        # 2. Prepare the input data
        # Get the last 60 days for the input sequence
        last_60_days = historical_data.iloc[-60:]
        
        # Use only the features the model was trained on
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        input_data = last_60_days[features]

        # 3. Scale the input data
        scaled_input_data = scaler.transform(input_data)
        
        # Reshape for the LSTM model: [1, 60, 5]
        reshaped_input = np.reshape(scaled_input_data, (1, 60, 5))

        # 4. Make a prediction
        predicted_scaled_price = model.predict(reshaped_input)

        # 5. Inverse transform the prediction to get the actual price
        # Create a dummy array with 5 features to perform the inverse transform
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 3] = predicted_scaled_price[0, 0]  # Put prediction in the 'Close' price column (index 3)
        predicted_price = scaler.inverse_transform(dummy_array)[0, 3]

        # Get the last known closing price for context
        last_known_price = historical_data['Close'].iloc[-1]

        # Log parameters and metrics to MLflow
        mlflow.log_param("ticker", "AMZN")
        mlflow.log_param("sequence_length", 60)
        mlflow.log_metric("last_known_price", round(last_known_price, 2))
        mlflow.log_metric("predicted_price", round(predicted_price, 2))
        
        # Log the model with MLflow (optional, can be done once after training)
        if not os.path.exists("mlflow_model"):
            mlflow.keras.log_model(model, "amazon_lstm_model", registered_model_name="AmazonStockPredictor")
            os.makedirs("mlflow_model", exist_ok=True) # Create a dummy folder to prevent re-logging

        # 6. Prepare and return the response
        response = {
            "model_used": "best_model.h5",
            "last_known_close_price": round(last_known_price, 2),
            "predicted_next_day_close_price": round(predicted_price, 2)
        }

        return response

# --- Run the App ---

if __name__ == '__main__':
    # This allows you to run the API directly from the script
    uvicorn.run(app, host="0.0.0.0", port=8000)