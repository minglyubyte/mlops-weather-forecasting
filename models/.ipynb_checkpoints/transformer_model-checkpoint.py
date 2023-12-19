import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

# Define the Transformer model
class WeatherTransformer(nn.Module):
    def __init__(self, input_size, num_layers, nhead, dim_feedforward, max_seq_length):
        super(WeatherTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, dim_feedforward))
        transformer_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(dim_feedforward, 1)  # Output a single value
        self.input_size = input_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x[:, -1, :].squeeze(-1)  # Take the output of the last time step