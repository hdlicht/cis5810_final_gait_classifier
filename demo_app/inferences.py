from lstm_model import LSTMModel

import numpy as np
import pandas as pd
import torch

def load_mp_model(model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=66, hidden_size=256, num_layers=3, output_size=12)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    return model

def load_dino_model(model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=768, hidden_size=128, num_layers=2, output_size=12)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    return model

def predict(model, input_data):
    # Convert input data to a PyTorch tensor
    class_names = ['abnormal','antalgic','cerebral palsy','exercise','inebriated','myopathic','normal','parkinsons','pregnant','prosthetic','stroke','style']
    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    else:
        input_data = input_data.float()
    input_tensor = input_data.unsqueeze(0)  # add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        # detach the output tensor from the current graph to prevent gradients from being calculated
        output = output.detach()
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
        confidence = confidence[predicted.item()]
        # convert to numpy array
        confidence = confidence.numpy()
        print(f"Predicted class: {predicted_class}, Confidence: {confidence.item()}")
        return predicted_class, confidence

