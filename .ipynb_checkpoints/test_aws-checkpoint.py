from pydantic import BaseModel

class PredictionParams(BaseModel):
    experiment_id: str
    experiment_name: str
    model_name: str
    version: int
    scaler_path: str

    class Config:
        protected_namespaces = ()
    
PredictionParams_data = {
    "experiment_id": "",
    "experiment_name": "mlops-weather-forecasting",
    "model_name": "weather_transformer",
    "version": 1,
    "scaler_path": "s3://leomlops/data_scaler.pkl"
}

prediction_params = PredictionParams(**PredictionParams_data)
import requests
url = "http://ec2-3-145-196-99.us-east-2.compute.amazonaws.com:8000/predict"
    
response = requests.post(url, json=prediction_params.model_dump())
print(response.status_code)
print(response.text)