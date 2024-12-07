import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


def init_dino_model():
    # Load DINO model from Hugging Face (e.g., "facebook/dino-vitb16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = AutoModel.from_pretrained("facebook/dino-vitb16").to(device)
    dino_model.eval()
    print("DINO model loaded.")
    return dino_model

def get_dino_features(model, frames):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    batch = []

    for frame in frames:
        input_tensor = transform(frame)
        batch.append(input_tensor)

    batch_tensor = torch.stack(batch).to(device)
    with torch.no_grad():
        output = model(batch_tensor)
        features = output.last_hidden_state[:, 0, :]  # [CLS] token features for each image
        features = features.detach().cpu()  # Move to CPU if needed
        return features
