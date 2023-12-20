import pandas as pd
import numpy as np
import datetime
import pytz
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text
from .config import get_config
# Define the log file path here
log_file_path = "output.txt"

# Redirect sys.stdout to the log file
sys.stdout = open(log_file_path, "w")

def unix_timestamp_to_datetime(timestamp):
    # Convert Unix time to UTC datetime
    utc_time = datetime.datetime.utcfromtimestamp(timestamp)

    # Use pytz to define the PST timezone
    pst_tz = pytz.timezone('America/Los_Angeles')

    # Convert UTC time to PST
    pst_time = utc_time.replace(tzinfo=pytz.utc).astimezone(pst_tz)

    # Format the date in Y-M-D format
    formatted_date = pst_time.strftime("%Y-%m-%d")

    return formatted_date

def timeseries_data_generator(X, y, length):
    features = []
    target = []
    for i in range(X.shape[0] - length):
        features.append(X[i:i+length])
        target.append(y[i+length])
    return np.array(features), np.array(target)

def data_transform(data_path):
    df = pd.read_csv(data_path)
    df_selected = df[['city_name','dt', 'temp','dew_point','feels_like','humidity']]
    df_selected['dt'] = df_selected['dt'].apply(unix_timestamp_to_datetime)
    df_selected = df_selected.groupby(by = ['city_name','dt']).mean().reset_index()
    return df_selected

def scaler_inverse_transform(scaler, X):
    matrix = []
    for x in X:
        matrix.append([x,0,0,0])
    inverse_transformed_matrix = scaler.inverse_transform(matrix)
    output = [i[0] for i in inverse_transformed_matrix]
    return output

def scaler_transform(scaler, X):
    matrix = []
    for x in X:
        matrix.append([x,0,0,0])
    transformed_matrix = scaler.transform(matrix)
    output = [i[0] for i in transformed_matrix]
    return output

def prepare_input(end_date, scaler, table_name = 'weather_data_original_mlops'):
    db_url = get_config()
    engine = create_engine(db_url)
    connection = engine.connect()
    query = text(f"SELECT * FROM {table_name} WHERE dt >= ('{end_date}'::date - INTERVAL '29 days') AND dt <= '{end_date}' ORDER by dt;")
    df = pd.read_sql(query, connection)
    df = df[['temp','dew_point','feels_like','humidity']]
    df = scaler.transform(df)
    return np.array(df)