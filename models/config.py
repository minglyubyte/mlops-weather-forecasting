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
    #db_url = f'postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:5432/{PG_DATABASE}'
    #print(db_url)
    #db_url = "postgresql+psycopg2://leo:Lm%40132465798@postgresql-db:5432/leo"
    db_url = "postgresql://leo:Lm%40132465798@0.0.0.0:5432/leo"
    
    return db_url