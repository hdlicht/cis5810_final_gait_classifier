import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

class LSTMwithAngle(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=0.01):
        super(LSTMwithAngle, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Define the LSTM layer and output layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + 4, output_size)
        self.loss_fn = nn.CrossEntropyLoss()  # Or use nn.MSELoss for regression tasks

    def forward(self, x, angle):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.cat((out, angle), dim=-1)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y, _, angle = batch
        y_hat = self(x, angle)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, angle = batch
        y_hat = self(x, angle)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y,  _, angle  = batch
        y_hat = self(x, angle)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
