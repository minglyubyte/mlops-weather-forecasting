import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from .transformer_model import WeatherTransformer
from data.data_preprocess import scaler_transform, scaler_inverse_transform

class Model():
    def __init__(self, RUN_ID, model_name, version, scaler):
        self.id = RUN_ID 
        self.model_name = model_name
        self.version = version
        self.scaler = scaler
        try:
            self.load_model()
        except Exception as e:
            print(f'Model not found initiating default model and training: {str(e)}')

    def save_model(self):
        mlflow.pytorch.log_model(
                pytorch_model = self.model,
                artifact_path = "pytorch-model",
                registered_model_name= self.model_name
        )

    def load_model(self):
        # Load a trained model from a pickle file
        self.model = mlflow.pytorch.load_model(model_uri=f"models:/{self.model_name}/{self.version}")

    def predict(self, X):
        X_tensor = torch.from_numpy(X).to(torch.float32)
        outputs = self.model(X_tensor)
        prediction = outputs.detach().numpy()
        inverse_transform_prediction = scaler_inverse_transform(self.scaler, prediction)
        return inverse_transform_prediction[0]


    def train_with_mlflow(self, train_loader, val_loader, patience_threshold, num_epochs, learning_rate, retrain = "False"):
        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params({
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "num_layers": self.model.num_layers,
                "nhead": self.model.nhead,
                "dim_feedforward": self.model.dim_feedforward
            })

            # Initialize the model, loss function, and optimizer
            if retrain:
                model = WeatherTransformer(self.model.input_size, self.model.num_layers, self.model.nhead, self.model.dim_feedforward, self.model.max_seq_length)
            else:
                model = self.model
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_epochs = 0
        
            # start training
            for epoch in range(num_epochs):
                # training phase
                model.train()
                training_loss = 0.0
                for batch_x, batch_y in train_loader:  # Number of batches
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y.squeeze(-1))  # Ensure target is of correct shape
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()
                mlflow.log_metric("train_rmse", (training_loss / len(train_loader)) ** 0.5)
        
                # validation phase
                model.eval()
                best_validation_loss = float('inf')
                validation_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        validation_loss += loss.item()

                # Calculate average validation loss
                average_validation_loss = validation_loss / len(val_loader)
                mlflow.log_metric("validation_rmse", (average_validation_loss) ** 0.5)

                # Check for early stopping
                if average_validation_loss < best_validation_loss:
                    best_validation_loss = average_validation_loss
                    patience = 0  # Reset patience counter
                    best_epochs = epoch
                    # save best torch model
                    torch.save(model.state_dict(), 'weather_transformer.pth')
                else:
                    patience += 1

                val_rmse = average_validation_loss ** 0.5
                print(f'Epoch [{epoch + 1}/{num_epochs}], Validation RMSE: {val_rmse:.4f}')

                # Check for early stopping condition
                if patience >= patience_threshold:
                    print(f'Early stopping at epoch {epoch + 1}')
                    mlflow.log_metric("epochs_trained", best_epochs)
                    # save best model by mlflow
                    model_state_dict = torch.load("weather_transformer.pth")
                    best_model = WeatherTransformer(self.model.input_size, self.model.num_layers, self.model.nhead, self.model.dim_feedforward, self.model.max_seq_length)
                    best_model.load_state_dict(model_state_dict)
                    mlflow.pytorch.log_model(
                        pytorch_model=best_model,
                        artifact_path="pytorch-model",
                        registered_model_name=model_name
                    )
                    break

            mlflow.log_metric("epochs_trained", best_epochs)
            # Load the model state dictionary
            model_state_dict = torch.load("weather_transformer.pth")
            best_model = WeatherTransformer(self.model.input_size, self.model.num_layers, self.model.nhead, self.model.dim_feedforward, self.model.max_seq_length)
            best_model.load_state_dict(model_state_dict)
            mlflow.pytorch.log_model(
                pytorch_model=best_model,
                artifact_path="pytorch-model",
                registered_model_name=self.model_name
            )
            mlflow.end_run()