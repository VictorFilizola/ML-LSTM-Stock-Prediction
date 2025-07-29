import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import mlflow
import mlflow.keras
import pickle
import os
import json
from datetime import datetime
import time

# --- FastAPI App and MLflow Setup ---

tags_metadata = [
    {
        "name": "Amazon Stock Information & Prediction",
        "description": "Endpoints for fetching historical data and predicting future Amazon (AMZN) stock closing prices."
    }
]

app = FastAPI(
    title="Amazon Stock Prediction API",
    description="An API to predict future closing prices of Amazon (AMZN) stock using an LSTM model.",
    version="2.0",
    openapi_tags=tags_metadata
)

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("amazon_stock_predictor")

# --- Load Model and Scaler ---

MODEL_PATH = "best_model.h5"
SCALER_PATH = "scaler.pkl"

try:
    model = load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# --- Helper Function for Robust Data Fetching ---

def fetch_data_with_retry(ticker_symbol: str, period: str, retries: int = 3, delay: int = 2):
    """
    Fetches historical data from yfinance with a retry mechanism.
    """
    for i in range(retries):
        try:
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=period, actions=False)
            if not data.empty:
                return data
            print(f"Attempt {i + 1}/{retries}: No data returned for {ticker_symbol}. Retrying in {delay} seconds...")
        except Exception as e:
            print(f"Attempt {i + 1}/{retries}: An exception occurred: {e}. Retrying in {delay} seconds...")
        time.sleep(delay)
    return pd.DataFrame() # Return empty DataFrame if all retries fail

# --- API Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to the Amazon Stock Prediction API"}

@app.get("/model-info", tags=["Amazon Stock Information & Prediction"])
async def model_info():
    """Provides metadata about the machine learning model being used."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found.")
    
    model_timestamp = os.path.getmtime(MODEL_PATH)
    model_last_modified = datetime.fromtimestamp(model_timestamp).strftime('%Y-%m-%d %H:%M:%S')

    return {
        "model_file": MODEL_PATH,
        "scaler_file": SCALER_PATH,
        "model_last_modified_utc": model_last_modified,
        "features_used": ['Open', 'High', 'Low', 'Close', 'Volume']
    }

@app.get("/history", tags=["Amazon Stock Information & Prediction"])
async def get_history(
    period: str = Query("3mo", description="The period of historical data to fetch.", enum=["1mo", "3mo", "6mo", "1y", "5y"])
):
    """Fetches recent historical stock data for AMZN."""
    historical_data = fetch_data_with_retry("AMZN", period=period)

    if historical_data.empty:
        raise HTTPException(status_code=404, detail="Could not fetch historical data for AMZN after multiple attempts.")

    # Convert timestamp index to string for JSON serialization
    historical_data.index = historical_data.index.strftime('%Y-%m-%d')
    return historical_data.to_dict(orient="index")


@app.get("/predict/{days_to_predict}", tags=["Amazon Stock Information & Prediction"])
async def predict_dynamic(
    days_to_predict: int = Path(..., gt=0, le=90, description="The number of future business days to predict (1-90).")
):
    """
    Predicts the stock's closing price for a specified number of future days.
    Note: Prediction accuracy decreases as the forecast horizon increases.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler is not available. Server might be initializing.")

    with mlflow.start_run(run_name=f"{days_to_predict}-Day Forecast"):
        # 1. Fetch historical data with retries
        historical_data = fetch_data_with_retry("AMZN", period="90d")
        
        if len(historical_data) < 60:
            raise HTTPException(status_code=400, detail=f"Not enough historical data. Found {len(historical_data)} days, need 60.")

        # 2. Prepare the initial input sequence
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        last_60_days = historical_data[features].tail(60)
        scaled_input = scaler.transform(last_60_days)
        
        future_predictions_scaled = []
        current_sequence = scaled_input.tolist()

        # 3. Iteratively predict future prices
        for _ in range(days_to_predict):
            input_array = np.array([current_sequence])
            predicted_scaled_price = model.predict(input_array)[0, 0]
            future_predictions_scaled.append(predicted_scaled_price)
            
            new_row = current_sequence[-1][:]
            new_row[3] = predicted_scaled_price
            
            current_sequence.append(new_row)
            current_sequence.pop(0)

        # 4. Inverse transform predictions
        dummy_array = np.zeros((days_to_predict, len(features)))
        dummy_array[:, 3] = np.array(future_predictions_scaled).flatten()
        predicted_prices = scaler.inverse_transform(dummy_array)[:, 3]

        # 5. Prepare response with future dates
        last_known_price = historical_data['Close'].iloc[-1]
        last_date = historical_data.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict).strftime('%Y-%m-%d').tolist()
        
        predictions_with_dates = {date: round(price, 2) for date, price in zip(future_dates, predicted_prices)}

        # 6. Log to MLflow
        mlflow.log_param("ticker", "AMZN")
        mlflow.log_param("prediction_days", days_to_predict)
        mlflow.log_param("model_used", MODEL_PATH) # Logs the model file as a parameter
        mlflow.log_metric("last_known_price", round(last_known_price, 2))
        mlflow.log_dict(predictions_with_dates, f"{days_to_predict}_day_predictions.json")

        # Log the model artifact itself, but only on the first run to avoid clutter
        if not os.path.exists("mlflow_model_logged.flag"):
            mlflow.keras.log_model(model, "amazon_lstm_model", registered_model_name="AmazonStockPredictor")
            # Create a flag file to indicate the model has been logged
            with open("mlflow_model_logged.flag", "w") as f:
                f.write("logged")

        return {
            "prediction_type": f"{days_to_predict}-Day Forecast",
            "last_known_close_price": round(last_known_price, 2),
            "predicted_prices": predictions_with_dates
        }

# --- Run the App ---

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

