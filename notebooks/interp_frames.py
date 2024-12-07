import os
import pandas as pd
import torch

# Directory containing landmark CSV files
lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/all_features"
new_lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/all_features_interp"

# Create the new directory if it does not exist
os.makedirs(new_lm_dir, exist_ok=True)

for file in [x for x in os.listdir(lm_dir) if x.endswith('.pt')]:
    # Get the full CSV path
    file_path = os.path.join(lm_dir, file)
    # Read the CSV into a tensor
    features_tensor = torch.load(file_path)
    initial_len = features_tensor.size(0)
    # Sort the tensor by the frame number (column 1 in your original DataFrame)
    features_tensor = features_tensor[torch.argsort(features_tensor[:, 0])]

    # List to store interpolated rows
    new_rows = []

    # Iterate through rows in the tensor
    for i in range(1, features_tensor.size(0)):
        current_frame = features_tensor[i, 0]
        previous_frame = features_tensor[i - 1, 0]

        # Check for missing frames
        if current_frame - previous_frame > 1:
            for missing_frame in range(int(previous_frame) + 1, int(current_frame)):
                # Interpolate for missing frames
                interpolated_row = features_tensor[i - 1] + (
                    (features_tensor[i] - features_tensor[i - 1])
                    * (missing_frame - previous_frame) / (current_frame - previous_frame)
                )
                interpolated_row[0] = missing_frame  # Set the frame number
                new_rows.append(interpolated_row)

    # Add interpolated rows to the original tensor
    if new_rows:
        new_rows_tensor = torch.stack(new_rows)
        features_tensor = torch.cat([features_tensor, new_rows_tensor], dim=0)
        # Sort by frame number after adding interpolations
        features_tensor = features_tensor[torch.argsort(features_tensor[:, 0])]

        print(f"Processed {file}: {initial_len} -> {features_tensor.size(0)}")

    # Save the tensor to a new CSV file
    new_csv_path = os.path.join(new_lm_dir, file)
    pd.DataFrame(features_tensor.numpy()).to_csv(new_csv_path, header=False, index=False)
