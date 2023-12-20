from models.transformer_model import WeatherTransformer
from models.model import Model
from data.data_preprocess import timeseries_data_generator, scaler_inverse_transform, scaler_transform, prepare_input
from data.data_ingest import daily_data_feed
from data.config import get_config
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import mlflow
import datetime
import pickle
import torch
import pytz

# Define the log file path here
log_file_path = "output.txt"

# Redirect sys.stdout to the log file
sys.stdout = open(log_file_path, "w")

def predict(model_name, version, scaler_path, RUN_ID):
    # get daily feed first
    daily_data_feed("f6a721a84c31decebf1e30fa38585ae0")

    # load scaler
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # get next date for prediction
    pst = pytz.timezone('US/Pacific')
    today_time = datetime.datetime.now(pst)
    today_date = today_time.date()
    next_day_time = today_time + datetime.timedelta(days=1)
    next_day_date = next_day_time.date()

    db_url = get_config()
    engine = create_engine(db_url)
    connection = engine.connect()
    query = text(f"SELECT * FROM weather_data_original_mlops_prediction")
    df_in_db = pd.read_sql(query, connection)
    date_set = set(df_in_db['dt'])
    if next_day_date in date_set:
        print(f"Prediction already made for date {next_day_date}")
        predictions = df_in_db[df_in_db['dt'] == next_day_date]
        return list(predictions['dt'])[0], list(predictions['temp'])[0], True

    # prepare input data, select features for the past 30 days and scale it with scaler
    X = prepare_input(today_date,scaler)
    reshaped_X = X.reshape((1, X.shape[0],X.shape[1]))
    
    # load model and predict
    model = Model(RUN_ID, model_name, version, scaler)
    prediction = model.predict(reshaped_X)
    print(next_day_date, prediction)
    
    return next_day_date, prediction, False

def monitor(scaler):
    # check if we have runned the test performace checking already for today
    pst = pytz.timezone('US/Pacific')
    today_time = datetime.datetime.now(pst)
    today_date = today_time.date()

    # get performance table data
    db_url = get_config()
    engine = create_engine(db_url)
    connection = engine.connect()
    
    query = text(f"SELECT * FROM weather_data_original_mlops_performance")
    df_in_db_performance = pd.read_sql(query, connection)
    date_set = set(df_in_db_performance['dt'])

    if today_date in date_set:
        performance = df_in_db_performance[df_in_db_performance['dt'] == today_date]
        print("Alreay ran the performance checking for date", today_date)
        print(performance)
        return list(performance['dt'])[0], list(performance['rmse'])[0], list(performance['count'])[0], True
    
    # get test data
    query = text(f"SELECT * FROM weather_data_original_mlops_test")
    df_in_db_test = pd.read_sql(query, connection)

    # get prediction data
    query = text(f"SELECT * FROM weather_data_original_mlops_prediction")
    df_in_db_predict = pd.read_sql(query, connection)

    # join tables
    result_left = df_in_db_predict.merge(df_in_db_test, on="dt", how="left")
    result_left = result_left.dropna()

    # get rmse from prediction and ground truth label
    x = scaler_transform(scaler, result_left['temp_x'].tolist())
    y = scaler_transform(scaler, result_left['temp_y'].tolist())
    rmse = np.sqrt(((np.array(x) - np.array(y)) ** 2).mean())
    
    return today_date, rmse, len(x), False



