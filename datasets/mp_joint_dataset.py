import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from joint_angles import calculate_angles

def gaussian_kernel(kernel_size, sigma):
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()  # Normalize
    return kernel.view(1, 1, -1) 

def preprocess_fft(signal, target_length):
    # Pad or truncate to make signal length consistent
    signal_length = signal.size(0)
    if signal_length < target_length:
        # Zero-pad shorter sequences
        padded_signal = F.pad(signal, (0, 0, 0, target_length - signal_length))
    else:
        # Truncate longer sequences
        padded_signal = signal[:target_length]
    return padded_signal

def normalize_signal(signal):
    # Normalize the signal
    signal = (signal - signal.mean()) / signal.std()
    return signal

def filter_signal(features):
    kernel = gaussian_kernel(5, 1.0)
    features = features.unsqueeze(0)
    features = features.permute(2, 0, 1)  # Shape: [batch_size, num_channels, signal_length] --> [1, 66, 151]
    features = F.conv1d(features, kernel, padding=2)
    features = features.permute(1, 2, 0)
    features = features.squeeze(0)
    return features

def get_fft_features(features, target_length, max_freq):
# Perform 120-point FFT on the features
    print(f"Beginning {features.shape}")
    features = preprocess_fft(features, target_length)
    print(f"After preprocess {features.shape}")
    fft = torch.fft.fft(features, dim=0)
    fft = torch.abs(fft)

    freqs = torch.fft.fftfreq(target_length, d=1/30)  # Frequency bins
    freq_mask = (freqs < max_freq) & (freqs >= 0.0)

    fft = fft[freq_mask, :]
    fft = fft.flatten()
    fft = normalize_signal(fft)

    return fft

def process_clip_for_inference(mp_features, dino_features, sequence_length, downsample_factor):
    total_fps = sequence_length * downsample_factor
    data = []
    
    # Preprocess and normalize
    mp_features = normalize_signal(mp_features)
    mp_features = filter_signal(mp_features)

    # Process chunks
    for i in range(0, len(mp_features), total_fps):
        chunk_mp = mp_features[i:min(len(mp_features), i + total_fps)]

        if len(chunk_all) < total_fps:
            chunk_all = torch.cat([chunk_all, torch.zeros(total_fps - len(chunk_all), chunk_all.size(1))])
        
        fft_features = get_fft_features(chunk_mp, total_fps, max_freq=1)
        
        for offset in range(downsample_factor):
            downsampled_chunk = chunk_all[offset::downsample_factor]
            
            if len(downsampled_chunk) < sequence_length:
                downsampled_chunk = torch.cat([downsampled_chunk, torch.zeros(sequence_length - len(downsampled_chunk), downsampled_chunk.size(1))])
            
            data.append((downsampled_chunk, fft_features))
    
    return data


class MpJointDataset(Dataset):
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
            files = [f for f in os.listdir(self.feature_dir) if f.startswith(clip)]
            for file_name in files:
                file_path = os.path.join(self.feature_dir, file_name)
                df = pd.read_csv(file_path)  # Load the CSV (even though it's named .pt)
                features = torch.tensor(df.values, dtype=torch.float32) 

                label = self.label_mapping.get(clip, None)
                angle = torch.tensor(self.angle_to_onehot[self.angle_mapping.get(clip, None)], dtype=torch.float32)

                # Extract and normalize Mediapipe features
                mp_features = features[:, 1:70]
                joint_angles = calculate_angles(mp_features)
                joint_angles = normalize_signal(joint_angles)
                joint_angles = filter_signal(joint_angles)
                # get rid of the finger points
                mp_features = torch.cat((mp_features[:, :21], mp_features[:, 39:]), dim=1)
                mp_features = normalize_signal(mp_features)
                mp_features = filter_signal(mp_features)

                features = torch.cat([mp_features, joint_angles], dim=1)

                # Create chunks of x*y frames and compute the FFT for each chunk
                total_fps = self.sequence_length * self.downsample_factor
                for i in range(0, len(mp_features), total_fps):
                    chunk_joints = joint_angles[i:min(len(joint_angles), i + total_fps)]
                    chunk_all = features[i:min(len(features), i + total_fps)]
                    
                    # Ensure chunk is of length x*y (pad if necessary)
                    if len(chunk_all) < total_fps:
                        chunk_all = torch.cat([chunk_all, torch.zeros(total_fps - len(chunk_all), chunk_all.size(1))])
                    # Generate FFT features for this x*y-frame chunk
                    fft_features = get_fft_features(joint_angles, total_fps, max_freq=3)  # Adjust max_freq if necessary

                    # Create y downsampled chunks of y with different offsets
                    for offset in range(self.downsample_factor):  # 3 different offsets
                        downsampled_chunk = chunk_all[offset::self.downsample_factor]
                        
                        # Ensure each downsampled chunk is of length x
                        if len(downsampled_chunk) < self.sequence_length:
                            downsampled_chunk = torch.cat([downsampled_chunk, torch.zeros(self.sequence_length - len(downsampled_chunk), downsampled_chunk.size(1))])

                        self.data.append((downsampled_chunk, fft_features, label, clip, angle))



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