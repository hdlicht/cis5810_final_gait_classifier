import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from collections import Counter
from image_dataset import ImageSequenceDataset
from cnn3d_model import CNN3D
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


def log_gpu_memory(step):
    """Logs GPU memory usage at a specific step."""
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1024**3   # Convert to GB
    print(f"[{step}] GPU memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Load annotations
annotation_path = "gcs/aaaaa/code/updated_dataset.pkl"
df = pd.read_pickle(annotation_path)
clip_label = df[['seq', 'gait_pat']].drop_duplicates()
clip_angle = df[['seq', 'cam_view']].drop_duplicates()
label_mapping = {row['seq']: row['gait_pat'] for idx, row in clip_label.iterrows()}
angle_mapping = {row['seq']: row['cam_view'] for idx, row in clip_angle.iterrows()}

# Load all sequence directories
image_dir = "gcs_images/gavd_frame_images"
sequence_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)) and d.startswith('c')]
labels = [label_mapping[seq] for seq in sequence_dirs]

# Count occurrences of each label
label_counts = Counter(labels)

# Keep only classes with at least 2 samples
valid_classes = {label for label, count in label_counts.items() if count > 1}
filtered_dirs = [seq for seq, label in zip(sequence_dirs, labels) if label in valid_classes]
filtered_labels = [label for label in labels if label in valid_classes]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(filtered_labels)
# Create a mapping of labels to their encodings
label_to_encoding = {label: idx for idx, label in enumerate(label_encoder.classes_)}
encoding_to_label = {idx: label for label, idx in label_to_encoding.items()}
encoded_label_mapping = {idx: label_to_encoding[label] for idx, label in label_mapping.items()}

# Stratified split
train_dirs, temp_dirs, train_labels, temp_labels = train_test_split(
    filtered_dirs, y_encoded, stratify=y_encoded, test_size=0.35, random_state=42
)
val_dirs, test_dirs, val_labels, test_labels = train_test_split(
    temp_dirs, temp_labels, stratify=temp_labels, test_size=0.5, random_state=42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create Datasets
train_dataset = ImageSequenceDataset(
    clips=train_dirs,
    image_dir=image_dir,
    label_mapping=encoded_label_mapping,
    angle_mapping=angle_mapping,
    sequence_length=20,
    downsample_factor=5,
    transform=transform
)
val_dataset = ImageSequenceDataset(
    clips=val_dirs,
    image_dir=image_dir,
    label_mapping=encoded_label_mapping,
    angle_mapping=angle_mapping,
    sequence_length=20,
    downsample_factor=5,
    transform=transform
)
test_dataset = ImageSequenceDataset(
    clips=test_dirs,
    image_dir=image_dir,
    label_mapping=encoded_label_mapping,
    angle_mapping=angle_mapping,
    sequence_length=20,
    downsample_factor=5,
    transform=transform
)

# Make data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize wandb
wandb.login(key='af968dcc6f77f75c5bcd9bfc39807a51bcecab9d')
wandb.init(project='3d-cnn-gait-analysis')

# Define the model (using the CNN3D class from your previous code)
model = CNN3D(num_classes=len(label_to_encoding.items()), learning_rate=0.001)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights.to(device)

# Use in loss
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Initialize the wandb logger
wandb.init(project='3d-cnn-gait-analysis')

# Track the best validation loss
best_val_loss = float('inf')
best_model_weights = None  # To hold the best model's weights

# Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for sequences, labels, angles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        angles = angles.to(device)

        optimizer.zero_grad()

        outputs = model(sequences, angles)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()


        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Log the training loss and accuracy to wandb
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels, angles in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            angles = angles.to(device)

            outputs = model(sequences, angles)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    # Log the validation loss and accuracy to wandb
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Save the model if the validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss

    # Save the model checkpoint
    torch.save(model.state_dict(), f"model_{epoch}.pth")
    print("Model saved with improved validation loss.")

# Testing the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for sequences, labels, angles in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        angles = angles.to(device)

        outputs = model(sequences, angles)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

# Log the test loss and accuracy to wandb
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Finish the wandb run
wandb.finish()

