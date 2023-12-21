from dotenv import load_dotenv
import os
import pandas as pd
import requests
import sys

# Load .env file
load_dotenv()

class PredictionParams(BaseModel):
    experiment_id: str
    experiment_name: str
    model_name: str
    version: int
    scaler_path: str
    
    class Config:
        protected_namespaces = ()

# Get the EC2 tracking server host from the environment variable
EC2_TRACKING_SERVER_HOST = os.getenv('EC2_TRACKING_SERVER_HOST')
EC2_ENDPOINT = f"http://{EC2_TRACKING_SERVER_HOST}:8000"

# Parameters for the RDS PostgreSQL instance
PG_HOST = os.getenv('PG_HOST')
PG_PORT = os.getenv('PG_PORT')
PG_DATABASE = os.getenv('PG_DATABASE')
PG_USER = os.getenv('PG_USER')
PG_PASSWORD = os.getenv('PG_PASSWORD')

def call_predict():
    PredictionParams_data = {
        "experiment_id": "",
        "experiment_name": "mlops-weather-forecasting",
        "model_name": "weather_transformer",
        "version": 1,
        "scaler_path": "data_scaler.pkl"
    }

    prediction_params = PredictionParams(**PredictionParams_data)
    response = requests.post(f'{EC2_ENDPOINT}/predict', json=prediction_params.model_dump())
    print(response.json())

def call_monitor():
    PredictionParams_data = {
        "experiment_id": "",
        "experiment_name": "mlops-weather-forecasting",
        "model_name": "weather_transformer",
        "version": 1,
        "scaler_path": "data_scaler.pkl"
    }

    prediction_params = PredictionParams(**PredictionParams_data)
    response = requests.post(f'{EC2_ENDPOINT}/monitor', json=prediction_params.model_dump())
    print(response.json())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python task_runner.py [predict|monitor|retrain]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'predict':
        call_predict()
    elif command == 'monitor':
        call_monitor()
    else:
        print("Invalid command. Use 'predict', 'monitor', or 'retrain'.")
