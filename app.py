from fastapi import FastAPI
from main import predict, monitor
from data.data_ingest import daily_data_feed, data_ingest
from data.data_preprocess import prepare_input
import datetime
import os
import mlflow
import uvicorn
import pandas as pd
import pickle
from pydantic import BaseModel

class PredictionParams(BaseModel):
    experiment_id: str
    experiment_name: str
    model_name: str
    version: int
    scaler_path: str

    class Config:
        protected_namespaces = ()

TRACKING_SERVER_HOST = os.environ.get("EC2_TRACKING_SERVER_HOST")
print(f"Tracking Server URI: '{TRACKING_SERVER_HOST}'")
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000") 

app = FastAPI()

@app.post("/predict")
def api_predict(params: PredictionParams):
    experiment_name = params.experiment_name
    experiment_id = params.experiment_id
    model_name = params.model_name
    version = params.version
    scaler_path = params.scaler_path
    
    today_time = datetime.datetime.now()
    today_date = today_time.strftime('%Y-%m-%d')

    print(f'Running a prediction for {today_date}, experiment name: {experiment_name} with model {model_name} v{version}.')
    if not experiment_id:
        try:
            print(f'Trying to create an experiment with name {experiment_name}')
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location="s3://leomlops")
        except:
            print(f'Experiment {experiment_name} exists')
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    run_name = f'pred_run_{today_date}'
    print(f'Run name: {run_name}')
    with mlflow.start_run(experiment_id=experiment_id ,run_name=run_name) as run: 
        next_day_date, prediction, alreay_predicted = predict(model_name, version, scaler_path, run_name)
        print(prediction, next_day_date)

    # ingest into prediction table
    if not alreay_predicted:
        data_ingest(data_dict = {'dt':next_day_date, 'temp':prediction}, date = next_day_date, type = "predict", table_name = "weather_data_original_mlops_prediction")
        return {"message": f"Prediction completed for {next_day_date}!", "predictions": prediction, "date": next_day_date}
    else:
        return {"message": f"Prediction already made for {next_day_date}!", "predictions": prediction, "date": next_day_date}

@app.post("/monitor")
def api_monitor(params: PredictionParams):
    experiment_name = params.experiment_name
    experiment_id = params.experiment_id
    model_name = params.model_name
    version = params.version
    scaler_path = params.scaler_path

    today_time = datetime.datetime.now()
    today_date = today_time.strftime('%Y-%m-%d')

    print(f'Running a monitoring for {today_date}, experiment name: {experiment_name} with model {model_name} v{version}.')
    if not experiment_id:
        try:
            print(f'Trying to create an experiment with name {experiment_name}')
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location="s3://leomlops")
        except:
            print(f'Experiment {experiment_name} exists')
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    run_name = f'monitor_run_{today_date}'
    print(f'Run name: {run_name}')
    
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
        
    with mlflow.start_run(experiment_id=experiment_id ,run_name=run_name) as run: 
        dt, rmse, count, alreay_predicted = monitor(scaler)

    if not alreay_predicted:
        data_ingest(data_dict = {'dt':dt, 'rmse':rmse, 'count':count}, date = dt, type = "monitor", table_name = "weather_data_original_mlops_performance")
        return {"message": f"Monitoring completed for {dt}!", 'dt':dt, 'rmse':rmse, 'count':count}
    else:
        return {"message": f"Monitoring already performed for {dt}!", 'dt':dt, 'rmse':rmse, 'count':count}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


















    
