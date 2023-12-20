import mlflow

model = mlflow.pytorch.load_model(f"models:/weather_transformer/7")
mlflow.set_tracking_uri("http://ec2-3-145-196-99.us-east-2.compute.amazonaws.com:5000")
# Load the PyTorch model
loaded_model = model
print(mlflow.create_experiment("mlops-weather-forecasting", artifact_location="s3://leomlops"))
#id = mlflow.get_experiment_by_name("mlops-weather-forecasting").experiment_id

# with mlflow.start_run(experiment_id=id) as run:
#     registered_model_name = "weather_transformer"
#     mlflow.pytorch.log_model(
#         artifact_path=registered_model_name,
#         pytorch_model=loaded_model,
#         registered_model_name=registered_model_name
#     )