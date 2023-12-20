from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

def get_config():
    # Parameters for the RDS PostgreSQL instance
    PG_HOST = os.environ.get('PG_HOST')
    PG_PORT = os.environ.get('PG_PORT')
    PG_DATABASE = os.environ.get('PG_DATABASE')
    PG_USER = os.environ.get('PG_USER')
    PG_PASSWORD = os.environ.get('PG_PASSWORD')

    # Create the MySQL database connection string
    db_url = f'postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}'
    #print(db_url)
    db_url = f'postgresql+psycopg2://lyum:Password85330060@mlops-weather-forecasting.ch134robrffl.us-east-2.rds.amazonaws.com:5432/initial_db'
    
    return db_url