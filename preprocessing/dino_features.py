import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
import mediapipe as mp


# Define paths
root_dir = "data/gavd_dataset/frame_images"
output_dir = "data/gavd_dataset/masked_dino_features"
os.makedirs(output_dir, exist_ok=True)

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINO model from Hugging Face (e.g., "facebook/dino-vitb16")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vitb16")
dino_model = AutoModel.from_pretrained("facebook/dino-vitb16").to(device)
dino_model.eval()

# Load mediapipe selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Define batch size
batch_size = 32

# Extract features with batch processing
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"Processing sequence: {subdir}")
        sequence_features = []  
        batch = []

        for image_file in tqdm(sorted(os.listdir(subdir_path)), desc=f"Images in {subdir}"):
            image_path = os.path.join(subdir_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                # Perform selfie segmentation
                image_np = np.array(image)
                result = mp_selfie_segmentation.process(image_np)
                mask = result.segmentation_mask > 0.1
                image_np[mask] = [0, 0, 0]  # Set the masked region to black
                image = Image.fromarray(image_np)
                input_tensor = transform(image)
                batch.append(input_tensor)
            except Exception as e:
                print(f"Warning: Failed to process {image_file} - {e}")
                continue

            if len(batch) == batch_size:
                batch_tensor = torch.stack(batch).to(device)
                with torch.no_grad():
                    output = dino_model(batch_tensor)
                    features = output.last_hidden_state[:, 0, :]  # [CLS] token features for each image
                    features = features.cpu()  # Move to CPU if needed
                    sequence_features.append(features)
                batch = []  # Reset batch

    

        # Process remaining images in the batch
        if batch:
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                output = dino_model(batch_tensor)
                # Extract the last hidden state and take the CLS token embeddings
                features = output.last_hidden_state[:, 0, :].cpu()  
                sequence_features.append(features)

        # Save features for the sequence
        if sequence_features:
            sequence_tensor = torch.cat(sequence_features)
            output_file = os.path.join(output_dir, f"{subdir}_features.pt")
            torch.save(sequence_tensor, output_file)
            print(f"Saved features for sequence '{subdir}' to {output_file}")
