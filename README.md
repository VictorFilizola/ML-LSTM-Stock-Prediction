Amazon (AMZN) Stock Price Prediction using LSTM
This project uses a Long Short-Term Memory (LSTM) neural network to predict the next day's closing price for Amazon (AMZN) stock. The project is composed of three main parts:

A Jupyter Notebook (main.ipynb) for data analysis, model training, and evaluation.

A FastAPI application (main.py) that serves the trained model as an API endpoint.

An MLflow integration for tracking prediction experiments.

Project Structure
.
├── main.ipynb              # Jupyter Notebook for model training and experimentation.
├── main.py                 # FastAPI application to serve the model.
├── client.py               # Example script to consume the prediction API.
├── best_model.h5           # The saved, trained Keras model.
├── scaler.pkl              # The saved scikit-learn scaler object.
├── requirements.txt        # Required Python libraries for the project.
└── README.md               # This documentation file.
Technologies Used
Python 3.8+

TensorFlow & Keras: For building and training the LSTM model.

Scikit-learn: For data preprocessing (MinMaxScaler).

Pandas: For data manipulation and analysis.

yfinance: To download historical stock market data.

FastAPI: To create and serve the prediction API.

Uvicorn: As the server to run the FastAPI application.

MLflow: For experiment tracking and logging predictions.

Jupyter Notebook: For interactive development and analysis.

Setup and Installation
Follow these steps to set up your local environment.

1. Clone the Repository
Bash

git clone https://github.com/VictorFilizola/ML-LSTM-Stock-Prediction.git
cd ML-LSTM-Stock-Prediction
2. Create a Virtual Environment (Recommended)
It's best practice to create a virtual environment to keep project dependencies isolated.

For Windows:

Bash

python -m venv venv
venv\Scripts\activate
For macOS/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
3. Install Required Libraries
Install all the necessary packages using the requirements.txt file.

Bash

pip install -r requirements.txt
How to Run the Project
This project involves three main components that need to be run in separate terminals: the MLflow server, the FastAPI server, and the client script.

Step 1: Train the Model (Optional)
The repository already includes a pre-trained model (best_model.h5) and a scaler (scaler.pkl). However, if you wish to retrain the model or experiment with different parameters, you can run the main.ipynb notebook in Jupyter.

Step 2: Start the MLflow Tracking Server
MLflow is used to log each prediction as an experiment.

Open a new terminal and navigate to the project directory. Run the following command to start the MLflow UI server:

Bash

mlflow ui
Keep this terminal running. You can view the MLflow dashboard in your browser at http://127.0.0.1:5000.

Step 3: Start the FastAPI Server
This server will load the model and expose the prediction endpoint.

Open a second terminal and navigate to the project directory. Run the following command to start the Uvicorn server:

Bash

uvicorn main:app --reload
The server will be running at http://127.0.0.1:8000. You can access the interactive API documentation at http://127.0.0.1:8000/docs.

Step 4: Run the Client to Get a Prediction
This script will send a request to the FastAPI server and print the result.

Open a third terminal and navigate to the project directory. Execute the client script:

Bash

python client.py
The terminal will display the predicted closing price for the next day. You can also check the MLflow UI to see the newly logged prediction run.

API Endpoint
GET /predict
Description: Fetches the last 60 days of AMZN stock data, preprocesses it, and returns the predicted closing price for the next trading day.

Method: GET

Success Response (200 OK):

JSON

{
  "model_used": "best_model.h5",
  "last_known_close_price": 185.57,
  "predicted_next_day_close_price": 186.25
}
Error Response:

JSON

{
  "error": "Not enough historical data available. Found only 58 days."
}