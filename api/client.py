import requests
import json

# The URL where your FastAPI application is running
api_url = "http://127.0.0.1:8000/predict"

print(f"Sending request to: {api_url}")

try:
    # Send a GET request to the /predict endpoint
    response = requests.get(api_url)

    # Print the HTTP status code
    print(f"Status Code: {response.status_code}")

    # Check if the request was successful
    if response.status_code == 200:
        # Print the JSON response from the API in a readable format
        prediction_data = response.json()
        print("\n--- Prediction Result ---")
        print(json.dumps(prediction_data, indent=2))
        print("-------------------------\n")
    else:
        # Print the error if the request failed
        print(f"\nError making request: {response.text}")

except requests.exceptions.ConnectionError as e:
    print("\n--- Connection Error ---")
    print("Could not connect to the API. Please ensure that:")
    print("1. The FastAPI server (main.py) is running.")
    print(f"2. The server is accessible at {api_url}.")
    print("------------------------\n")

print("Thank you for using this API!")
