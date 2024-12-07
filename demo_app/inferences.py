from lstm_model import LSTMModel
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.lstm_fft_model import LSTM_FF_Model
from models.lstm_model import LSTMModel
from datasets.mega_dataset import process_clip_for_inference

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
num_classes = 4  # Get the number of classes from the dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

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

def load_mega_model(model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_FF_Model(input_size=837, hidden_size=128, num_layers=2, output_size=11, fft_input_size=276, fc_hidden_size=128)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    return model


def load_cam_classifier(model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_classes = 4 
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
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
    

def predict_mega_model(model, mp_features, dino_features, angle):
    prepared_data = process_clip_for_inference(mp_features, dino_features, 40, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Extract true labels, clip IDs, and predictions
    lstm_in = torch.stack([x[0] for x in prepared_data]).to(device)
    fft_in = torch.stack([x[1] for x in prepared_data]).to(device)

    angle_to_onehot = {'front': [1, 0, 0, 0], 'back': [0, 1, 0, 0], 'left side': [0, 0, 1, 0], 'right side': [0, 0, 0, 1]}
    one_hot = torch.tensor(angle_to_onehot[angle])
    angles_in = torch.stack([one_hot for _ in range(3)]).to(device)

    # Predict using the model
    with torch.no_grad():
        logits, features = model(lstm_in, fft_in, angles_in, extract_features=True)
        logits = logits.detach().cpu()
        features = features.detach().cpu()

    class_names = ['abnormal','antalgic','cerebral palsy','exercise','inebriated','myopathic','normal','parkinsons','prosthetic','stroke','style']
    output = torch.nn.functional.softmax(logits)
    row, col = torch.argmax(output).item() // output.size(1), torch.argmax(output).item() % output.size(1)
    predicted_class = class_names[col]
    confidence = output[row, col] * 100

    # Return the predicted class and confidence and the extracted features from the chunk with the highest confidence
    print(f"Predicted class: {predicted_class}, Confidence: {confidence.item()}")
    return predicted_class, confidence, features[row]

def predict_cam_view(model, images):
    class_labels = ['front','back','left side','right side']

    # Define the image preprocessing steps
    preprocess = transforms.Compose([             # Crop the center to 224x224
                transforms.ToTensor(),                      # Convert image to Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
            ])
    
    # Move the input batch to the same device as the model (if using GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_batch = torch.stack([preprocess(image.convert('RGB')) for image in images]).to(device)

    # Perform the inference
    with torch.no_grad():  # No need to track gradients during inference
        logits = model(input_batch)
    # Get the predicted class (class with the highest score)
    predicted_idx = torch.argmax(logits, dim=1)
    consensus_idx = torch.mode(predicted_idx).values.item()

    # Map the predicted index to the corresponding camera angle label
    predicted_angle = class_labels[consensus_idx]
        
    return predicted_angle


