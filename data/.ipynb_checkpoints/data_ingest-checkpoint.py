import pandas as pd
import numpy as np
import datetime
import requests
import pytz
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text
from .config import get_config
from .data_preprocess import unix_timestamp_to_datetime
# Define the log file path here
log_file_path = "output.txt"

# Redirect sys.stdout to the log file
sys.stdout = open(log_file_path, "w")

def daily_data_feed(api,orginal_table_name = 'weather_data_original_mlops', test_table_name = 'weather_data_original_mlops_test'):
    response = requests.get(f"https://api.openweathermap.org/data/3.0/onecall?lat=34.05&lon=-118.24&appid={api}")
    if response.status_code == 200:
        data = response.json()
        pst = pytz.timezone('US/Pacific')
        today_date = datetime.datetime.now(pst).strftime('%Y-%m-%d')

        #ran at 9 p.m. PST every day to get the average daily weather, create the dataframe to append
        weather_stats = data['current']
        city = 'Los Angeles'
        dt = unix_timestamp_to_datetime(weather_stats['dt'])
        temp = weather_stats['temp']
        dew_point = weather_stats['dew_point']
        feels_like = weather_stats['feels_like']
        humidity = weather_stats['humidity']
        if dt != today_date:
            print("dt:", dt)
            print("today date:", today_date)
            print("Date not correct!")
            return
        
        new_date_dict = pd.DataFrame({'city_name':city,'dt':dt,'temp':temp,'dew_point':dew_point,'feels_like':feels_like,'humidity':humidity},index=[0])
        
        db_url = get_config()
        engine = create_engine(db_url)
        connection = engine.connect()
        query = text(f"SELECT * FROM {orginal_table_name}")
        df_in_db = pd.read_sql(query, connection)
        date_set = set(df_in_db['dt'])
        today_date_datetime_object = datetime.datetime.strptime(today_date, "%Y-%m-%d").date()

        # checking if the weather within specific date is stored already
        if today_date_datetime_object not in date_set:
            new_date_dict.to_sql(orginal_table_name, engine, if_exists='append', index=False)
            new_date_dict.to_sql(test_table_name, engine, if_exists='append', index=False)
            print(f"Successfully Append {today_date} weather data to original and test table!")
        else:
            print(f"{today_date} weather data alreay appended to original and test table!")
        
    else:
        print("API Reponse not handled correct with resposne code:", response.status_code)

def data_ingest(data_dict, date, type, table_name):
    db_url = get_config()
    engine = create_engine(db_url)
    df = pd.DataFrame(data_dict, index = [0])
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Successfully ingest date {date} {type} into table {table_name}")





