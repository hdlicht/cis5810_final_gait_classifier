import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_FF_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fft_input_size, fc_hidden_size, dropout_rate = 0.2):
        super(LSTM_FF_Model, self).__init__()
        
        # LSTM Layer for processing time-domain sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)

        # Fully connected layer for processing FFT features
        self.fc_fft = nn.Linear(fft_input_size, fc_hidden_size)
        
        # Final classification layer
        self.fc_2 = nn.Linear(hidden_size + fc_hidden_size, hidden_size)  # Concatenate LSTM and FC outputs
        self.fc_out = nn.Linear(hidden_size + 4, output_size)  # Concatenate LSTM and FC outputs

        # Dropout for regularization
        self.dropout_rate = dropout_rate

        # Save LSTM configuration parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def attention(self, lstm_outputs):
        # Compute attention scores (shape: batch_size, seq_len)
        attention_scores = F.softmax(torch.mean(lstm_outputs, dim=2), dim=1)
        
        # Weighted sum of LSTM outputs based on attention scores
        attention_output = torch.bmm(attention_scores.unsqueeze(1), lstm_outputs).squeeze(1)  # Shape: (batch_size, hidden_size)
        return attention_output

    def forward(self, x_seq, fft_features, angle, extract_features=False):
        # Pass sequence through LSTM
        h0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        c0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        lstm_out, _ = self.lstm(x_seq, (h0, c0))

        # Use the output from the last LSTM timestep
        attention_output = self.attention(lstm_out)  # Shape: (batch_size, hidden_size)

        # Pass FFT features through fully connected layer
        fft_out = F.relu(self.fc_fft(fft_features))

        # Concatenate LSTM and FFT outputs
        combined = torch.cat((attention_output, fft_out), dim=1)
        
        # Apply dropout for regularization
        combined = F.dropout(combined,p=self.dropout_rate, training=self.training)
        
        # Penultimate step: reduce to 128 and add the angle vector
        output = F.relu(self.fc_2(combined))
        penultimate = torch.cat((output, angle), dim=-1)
        output = self.fc_out(penultimate)

        if extract_features:
            return output, penultimate
        else:
            return output