import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


# Directory containing landmark CSV files
new_lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/landmarks_new"

frame_count = []

# Iterate through the CSV files
for file in [x for x in os.listdir(new_lm_dir) if x.endswith('.pt')]:
    # Load the landmarks
    landmarks = torch.load(os.path.join(new_lm_dir, file))
    frames = landmarks.shape[0]
    frame_count.append(frames)

# plot histogram of frame counts as a bar chart
pd.Series(frame_count).hist(bins=20)
# describe
print(pd.Series(frame_count).describe())
plt.xlabel("Number of Frames")
plt.ylabel("Number of Sequences")
plt.title("Distribution of Sequence Lengths")
plt.show()


    