# Amazon (AMZN) Stock Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next day's closing price for Amazon (AMZN) stock. The project is composed of three main parts:

1. A **Jupyter Notebook** (`main.ipynb`) for data analysis, model training, and evaluation.
2. A **FastAPI application** (`main.py`) that serves the trained model as an API endpoint.
3. An **MLflow integration** for tracking prediction experiments.

---

## Project Structure

```
.
├── main.ipynb              # Jupyter Notebook for model training and experimentation
├── main.py                 # FastAPI application to serve the model
├── client_test.py          # Example script to consume the prediction API
├── best_model.h5           # The saved, trained Keras model
├── scaler.pkl              # The saved scikit-learn scaler object
├── requirements.txt        # Required Python libraries for the project
└── README.md               # This documentation file
```

---

## Technologies Used

* **Python 3.8+**
* **TensorFlow & Keras:** For building and training the LSTM model
* **Scikit-learn:** For data preprocessing (MinMaxScaler)
* **Pandas:** For data manipulation and analysis
* **yfinance:** To download historical stock market data
* **FastAPI:** To create and serve the prediction API
* **Uvicorn:** As the server to run the FastAPI application
* **MLflow:** For experiment tracking and logging predictions
* **Jupyter Notebook:** For interactive development and analysis

---

## Setup and Installation

Follow these steps to set up your local environment.

### 1. Clone the Repository

```bash
git clone https://github.com/VictorFilizola/ML-LSTM-Stock-Prediction.git
cd ML-LSTM-Stock-Prediction
```

### 2. Create a Virtual Environment (Recommended)

It's best practice to create a virtual environment to keep project dependencies isolated.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Libraries

Install all the necessary packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

This project involves three main components that need to be run in separate terminals: the MLflow server, the FastAPI server, and the client script.

### Step 1: Train the Model (Optional)

The repository already includes a pre-trained model (`best_model.h5`) and a scaler (`scaler.pkl`). However, if you wish to retrain the model or experiment with different parameters, you can run the `main.ipynb` notebook in Jupyter.

### Step 2: Start the MLflow Tracking Server

MLflow is used to log each prediction as an experiment.

1. Open a new terminal and navigate to the project directory.
2. Run the following command to start the MLflow UI server:

```bash
mlflow ui
```

Keep this terminal running. You can view the MLflow dashboard in your browser at http://127.0.0.1:5000.

### Step 3: Start the FastAPI Server

This server will load the model and expose the prediction endpoint.

1. Open a second terminal and navigate to the project directory.
2. Run the following command to start the Uvicorn server:

```bash
uvicorn main:app --reload
```

The server will be running at http://127.0.0.1:8000. You can access the interactive API documentation at http://127.0.0.1:8000/docs.

### Step 4: Run the Client to Get a Prediction

This script will send a request to the FastAPI server and print the result.

1. Open a third terminal and navigate to the project directory.
2. Execute the client script:

```bash
python client_test.py
```

The terminal will display the results from all API endpoints. You can also check the MLflow UI to see the newly logged prediction runs.

---

## API Endpoints

### `GET /predict/{days_to_predict}`

**Description:** Predicts the closing price for a specified number of future business days.

**Path Parameter:**
- `days_to_predict` (integer): The number of days to forecast (e.g., `/predict/7` for a 7-day forecast). Must be between 1 and 90.

**Success Response (200 OK):**
```json
{
  "prediction_type": "7-Day Forecast",
  "last_known_close_price": 185.57,
  "predicted_prices": {
    "2024-08-27": 186.25,
    "2024-08-28": 187.01,
    "...": "..."
  }
}
```

### `GET /history`

**Description:** Fetches recent historical stock data for AMZN.

**Query Parameter:**
- `period` (string, optional): The time period. Defaults to `3mo`. Allowed values: `1mo`, `3mo`, `6mo`, `1y`, `5y`.

**Example:** `http://127.0.0.1:8000/history?period=1mo`

**Success Response (200 OK):**
```json
{
  "2024-07-26": {
    "Open": 184.08,
    "High": 185.75,
    "Low": 183.53,
    "Close": 185.57,
    "Volume": 35834100
  },
  "...": "..."
}
```

### `GET /model-info`

**Description:** Provides metadata about the machine learning model being used.

**Success Response (200 OK):**
```json
{
  "model_file": "best_model.h5",
  "scaler_file": "scaler.pkl",
  "model_last_modified_utc": "2024-08-26 14:30:00",
  "features_used": ["Open", "High", "Low", "Close", "Volume"]
}
```

