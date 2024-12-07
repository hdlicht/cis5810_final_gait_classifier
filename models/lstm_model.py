import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=0.01):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Define the LSTM layer and output layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()  # Or use nn.MSELoss for regression tasks

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y,  _, _  = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # Shape: (batch_size, seq_len, hidden_size)

        # Attention weights
        attention_weights = F.softmax(torch.mean(lstm_out, dim=2), dim=1)  # Shape: (batch_size, seq_len)
        
        # Weighted sum of LSTM outputs
        attention_output = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # Shape: (batch_size, hidden_size)

        # Final output
        out = self.fc(attention_output)
        return out
    
class LSTMWithAttentionAndReg(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3, learning_rate=0.001):
        super(LSTMWithAttentionAndReg, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Optional: Batch normalization
        self.bn = nn.BatchNorm1d(hidden_size)

    def attention(self, lstm_outputs):
        # Compute attention scores (shape: batch_size, seq_len)
        attention_scores = F.softmax(torch.mean(lstm_outputs, dim=2), dim=1)
        
        # Weighted sum of LSTM outputs based on attention scores
        attention_output = torch.bmm(attention_scores.unsqueeze(1), lstm_outputs).squeeze(1)  # Shape: (batch_size, hidden_size)
        return attention_output

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))  # Shape: (batch_size, seq_len, hidden_size)

        # Apply dropout to LSTM outputs
        lstm_out = F.dropout(lstm_out, p=self.dropout_rate, training=self.training)

        # Attention mechanism
        attention_output = self.attention(lstm_out)  # Shape: (batch_size, hidden_size)

        # Apply batch normalization
        attention_output = self.bn(attention_output)

        # Fully connected layer
        out = self.fc(attention_output)  # Shape: (batch_size, output_size)
        return out

