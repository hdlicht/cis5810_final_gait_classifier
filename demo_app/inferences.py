from lstm_model import LSTMModel

import numpy as np
import pandas as pd
import torch

def load_mp_model(model_path):
    # Load the model
    model = LSTMModel(input_size=3, hidden_size=128, num_layers=2, output_size=12)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_dino_model(model_path):
    # Load the model
    model = LSTMModel(input_size=384, hidden_size=128, num_layers=2, output_size=12)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_data):
    # Convert input data to a PyTorch tensor
    class_names = ['abnormal','antalgic','cerebral palsy','exercise','inebriated','myopathic','normal','parkinsons','pregnant','prosthetic','stroke','style']
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        confidence, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        print(f"Predicted class: {predicted_class}, Confidence: {confidence.item()}")
        return predicted_class, confidence

