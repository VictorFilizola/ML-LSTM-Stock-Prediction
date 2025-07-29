import requests
import json
import time

# Base URL where your FastAPI application is running
BASE_URL = "http://127.0.0.1:8000"

def make_request(endpoint: str, description: str):
    """Sends a request to a specified endpoint and prints the response."""
    api_url = f"{BASE_URL}/{endpoint}"
    print(f"--- {description} ---")
    print(f"Sending request to: {api_url}")

    try:
        response = requests.get(api_url)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\nResponse:")
            print(json.dumps(data, indent=2))
        else:
            print(f"\nError: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Could not connect to the API.")
        print("Please ensure the FastAPI server (main.py) is running.")
    
    print("-" * (len(description) + 8) + "\n")
    time.sleep(1) # Small delay between requests

if __name__ == "__main__":
    # 1. Get model information
    make_request("model-info", "Fetching Model Information")

    # 2. Get 1 month of historical data
    make_request("history?period=1mo", "Fetching 1 Month of Historical Data")

    # 3. Get a prediction for the next single day
    make_request("predict/1", "Requesting 1-Day Forecast")

    # 4. Get a 7-day forecast
    make_request("predict/7", "Requesting 7-Day Forecast")
    
    # 5. Get a 30-day forecast
    make_request("predict/30", "Requesting 30-Day Forecast")

    print("Thank you for using this API!")
