import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from joint_angles import calculate_angles


class MediapipeDataset(Dataset):
    def __init__(self, clips, feature_dir, label_mapping, angle_mapping, sequence_length=30, downsample_factor=2):
        """
        Initialize the dataset with configurations.

        Args:
            feature_dir (str): Directory containing DINO feature files.
            label_mapping (dict): Mapping of clip IDs to labels.
            angle_mapping (dict): Mapping of clip IDs to camera angles.
            sequence_length (int): Length of each sequence chunk.
            downsample_factor (int): Factor to downsample sequences.
        """
        self.clips = clips
        self.feature_dir = feature_dir
        self.label_mapping = label_mapping
        self.angle_mapping = angle_mapping
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.angle_to_onehot = {'front': [1, 0, 0, 0], 'back': [0, 1, 0, 0], 'left side': [0, 0, 1, 0], 'right side': [0, 0, 0, 1],'other': [0, 0, 0, 0],'': [0, 0, 0, 0]}

        self.data = []  # Store processed sequences

        self._prepare_data()

    def _prepare_data(self):
        """
        Process the DINO feature files and prepare sequences for the dataset.
        """
        for clip in self.clips:
            file_name = f"{clip}_landmarks.csv"
            file_path = os.path.join(self.feature_dir, file_name)
            # Load DINO features and get metadata
            features_df = pd.read_csv(file_path, header = None)
            features_df = features_df.sort_values(by=1, key=lambda col: col.astype(float))
            features_df = features_df.iloc[:, 35:]
            features = torch.tensor(features_df.values.astype(float), dtype=torch.float32)
            label = self.label_mapping.get(clip, None)
            angle = torch.tensor(self.angle_to_onehot[self.angle_mapping.get(clip, None)], dtype=torch.float32)


            # Create downsampled sequences with multiple offsets
            downsampled_sequences = [features[offset::self.downsample_factor] for offset in range(self.downsample_factor)]
            for sequence in downsampled_sequences:
                # Split into chunks of sequence_length
                for i in range(0, len(sequence), self.sequence_length):
                    chunk = sequence[i:i + self.sequence_length]

                    # Check if the chunk contains NaNs
                    if torch.isnan(chunk).any():
                        continue  # Skip this chunk if it contains NaNs

                    # Pad if necessary
                    if len(chunk) < self.sequence_length:
                        chunk = torch.cat([chunk, torch.zeros(self.sequence_length - len(chunk), chunk.size(1))])

                    # Add the chunk to the data
                    self.data.append((chunk, label, clip, angle))


    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing a sequence, label, clip ID, and angle.
        """
        return self.data[idx]
