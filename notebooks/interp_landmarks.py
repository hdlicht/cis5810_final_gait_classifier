import os
import pandas as pd
import torch

# Directory containing landmark CSV files
lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/landmarks"
new_lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/landmarks_new"

# Create the new directory if it does not exist
os.makedirs(new_lm_dir, exist_ok=True)

for file in [x for x in os.listdir(lm_dir) if x.endswith('.csv')]:
    # Get the clip name
    clip = file.split('_')[0]
    # Get the full CSV path
    csv_path = os.path.join(lm_dir, file)
    # Read the CSV
    features_df = pd.read_csv(csv_path, header=None)
    
    # Sort the dataframe by frame number
    features_df = features_df.sort_values(by=1, key=lambda col: col.astype(float))
    features_df = features_df.drop(features_df.columns[[0] + list(range(5, 35))], axis=1)

    # Reset the index
    features_df.reset_index(drop=True, inplace=True)
    # Drop the seq column (assuming the first column is the frame number)
    features_df = features_df.iloc[:, 1:]
    initial_len = len(features_df)
    
    # Get the frame numbers
    new_rows = []  # List to store interpolated rows

    # Iterate through rows in the DataFrame
    for index in range(1, len(features_df)):
        current_frame = features_df.iloc[index, 0]
        previous_frame = features_df.iloc[index - 1, 0]
        
        # Check for missing frames
        if current_frame - previous_frame > 1:
            # Interpolate for all missing frames
            for missing_frame in range(int(previous_frame) + 1, int(current_frame)):
                # Interpolate values
                interpolated_row = features_df.iloc[index - 1] + (
                    (features_df.iloc[index] - features_df.iloc[index - 1])
                    * (missing_frame - previous_frame) / (current_frame - previous_frame)
                )
                interpolated_row[0] = missing_frame  # Set the frame number
                new_rows.append(interpolated_row)

    # Add new rows to the DataFrame
    if new_rows:
        interpolated_df = pd.DataFrame(new_rows, columns=features_df.columns)
        features_df = pd.concat([features_df, interpolated_df], ignore_index=True)
        # Sort by frame number again after adding interpolations
        features_df = features_df.sort_values(by=0).reset_index(drop=True)

        print(f"Processed {file}: {initial_len} -> {len(features_df)} rows (with interpolation)")

    # # save to a new CSV file
    # new_csv_path = os.path.join(new_lm_dir, file)
    # features_df.to_csv(new_csv_path, header=False, index=False)

