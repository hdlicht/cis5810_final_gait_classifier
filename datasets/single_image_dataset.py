import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import numpy as np

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_df, image_dir, transform=None):

        self.image_dir = image_dir
        self.image_df = image_df
        self.transform = transform 
        print(f"Dataset loaded with {len(self.image_df)} images")

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        """
        Load and return a single image
        """
        # get the index row of the dataframe
        row = self.image_df.iloc[idx]
        seq = row['seq']
        frame = row['frame_num']
        # encode label to integer
        view = row['cam_view']
        labels = {'front': 0, 'back': 1, 'left side': 2, 'right side': 3}
        label = labels[view]
        # load image
        image_path = os.path.join(self.image_dir, seq, f'{seq}_frame_{frame}.jpg')
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label
