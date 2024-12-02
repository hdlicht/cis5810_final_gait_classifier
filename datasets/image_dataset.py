import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import numpy as np

class ImageSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, clips, image_dir, label_mapping, angle_mapping, 
                 sequence_length=30, downsample_factor=2, transform=None):
        """
        Initialize the dataset by precomputing subsequences of image paths.

        Args:
            clips (list): List of clip names to include in the dataset.
            image_dir (str): Base directory containing the image folders for each clip.
            label_mapping (dict): Mapping from clip name to label.
            angle_mapping (dict): Mapping from clip name to camera angle.
            sequence_length (int): Number of images in each subsequence.
            downsample_factor (int): Factor to downsample frames (e.g., every 2nd frame).
            transform (callable): Transformation to apply to images.
        """
        self.image_dir = image_dir
        self.label_mapping = label_mapping
        self.angle_mapping = angle_mapping
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.transform = transform 
        self.angle_to_onehot = {'front': [1, 0, 0, 0], 'back': [0, 1, 0, 0], 'left side': [0, 0, 1, 0], 'right side': [0, 0, 0, 1],'other': [0, 0, 0, 0],'': [0, 0, 0, 0]}


        # Precompute subsequences
        self.subsequences = []
        self.labels = []
        self.angles = []

        for clip in clips:
            clip_path = os.path.join(image_dir, clip)
            if not os.path.exists(clip_path):
                continue

            # Retrieve image files and sort them
            image_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])
            image_paths = [os.path.join(clip_path, img) for img in image_files]

            # Downsample frames
            downsampled_sequences = [
                image_paths[offset::self.downsample_factor] for offset in range(self.downsample_factor)
            ]

            # Generate subsequences
            for sequence in downsampled_sequences:
                for i in range(0, len(sequence), self.sequence_length):
                    chunk = sequence[i:i + self.sequence_length]

                    # Pad if necessary
                    if len(chunk) < self.sequence_length:
                        padding = [None] * (self.sequence_length - len(chunk))
                        chunk.extend(padding)

                    self.subsequences.append(chunk)
                    self.labels.append(label_mapping.get(clip, None))
                    angle_onehot = torch.tensor(self.angle_to_onehot[angle_mapping.get(clip, None)], dtype=torch.float32)
                    self.angles.append(angle_onehot)

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        """
        Build an image tensor for the given subsequence.

        Args:
            idx (int): Index of the subsequence.

        Returns:
            dict: Containing the subsequence tensor, label, and metadata.
        """
        image_paths = self.subsequences[idx]
        label = self.labels[idx]
        angle = self.angles[idx]

        # Load images and apply transformations
        images = []
        for img_path in image_paths:
            if img_path is not None and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            else:
                # Use a zero tensor for padding
                image = torch.zeros(3, 224, 224)
            images.append(image)

        images = torch.stack(images)  # Shape: (sequence_length, 3, 224, 224)

        return images, label, angle
        
