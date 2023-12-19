import requests
import pytz
import datetime
import pandas as pd
from pydantic import BaseModel
from data.config import get_config
from sqlalchemy import create_engine, text

class PredictionParams(BaseModel):
    experiment_id: str
    experiment_name: str
    model_name: str
    version: int
    scaler_path: str
    
    class Config:
        protected_namespaces = ()

def test_predict_endpoint():
    url = "http://0.0.0.0:8000/predict"

    PredictionParams_data = {
    "experiment_id": "0",
    "experiment_name": "123456",
    "model_name": "weather_transformer",
    "version": 7,
    "scaler_path": "data_scaler.pkl"
    }

    prediction_params = PredictionParams(**PredictionParams_data)
    response = requests.post(url, json=prediction_params.model_dump())
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"

    data = response.json()
    print(data)

    db_url = "postgresql+psycopg2://leo:Lm%40132465798@0.0.0.0:5432/leo"
    engine = create_engine(db_url)
    connection = engine.connect()
    
    # get next date for prediction
    pst = pytz.timezone('US/Pacific')
    today_time = datetime.datetime.now(pst)
    today_date = today_time.date()
    next_day_time = today_time + datetime.timedelta(days=1)
    next_day_date = next_day_time.date()
    
    query = text(f"SELECT * FROM weather_data_original_mlops_test")
    df_in_db_test = pd.read_sql(query, connection)
    date_set_test = set(df_in_db_test['dt'])

    query = text(f"SELECT * FROM weather_data_original_mlops_prediction")
    df_in_db_pred = pd.read_sql(query, connection)
    date_set_pred = set(df_in_db_pred['dt'])
    
    # Assert that the data are correctly ingested by checking the date
    assert today_date in date_set_test, f"Today {today_date} weather data is not ingested!"
    assert next_day_date in  date_set_pred, f"Prediction for {next_day_date} is not ingested!"
    return

def test_monitor_endpoint():
    url = "http://0.0.0.0:8000/monitor"
    PredictionParams_data = {
    "experiment_id": "0",
    "experiment_name": "123456",
    "model_name": "weather_transformer",
    "version": 7,
    "scaler_path": "data_scaler.pkl"
    }

    prediction_params = PredictionParams(**PredictionParams_data)
    response = requests.post(url, json=prediction_params.model_dump())
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"

    data = response.json()
    print(data)
    
    db_url = "postgresql+psycopg2://leo:Lm%40132465798@0.0.0.0:5432/leo"
    engine = create_engine(db_url)
    connection = engine.connect()
    
    # get next date for prediction
    pst = pytz.timezone('US/Pacific')
    today_time = datetime.datetime.now(pst)
    today_date = today_time.date()
    
    query = text(f"SELECT * FROM weather_data_original_mlops_performance")
    df_in_db_performance = pd.read_sql(query, connection)
    date_set_perf = set(df_in_db_performance['dt'])

    # Assert that the performance monitoring for today is ingested.
    assert today_date in date_set_perf, f"Performance monitoring for {today_date} is not ingested!"

if __name__ == "__main__":
    test_predict_endpoint()
    test_monitor_endpoint()