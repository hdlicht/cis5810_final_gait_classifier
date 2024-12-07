import os
import pandas as pd
import torch

# Directory containing landmark CSV files
lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/landmarks"
new_lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/landmarks_newer"

# Create the new directory if it does not exist
os.makedirs(new_lm_dir, exist_ok=True)

# Initialize lists for tensors and metadata
tensor_list = []
metadata_list = []
keep_frames = pd.DataFrame()
keep_frames['seq'] = []
keep_frames['frame_num'] = []
keep_frames['keep'] = []

# Iterate through the CSV files
for file in [x for x in os.listdir(lm_dir) if x.endswith('.csv')]:
    # Get the clip name
    clip = file.split('_')[0]
    # Get the full CSV path
    csv_path = os.path.join(lm_dir, file)
    # Read the CSV
    features_df = pd.read_csv(csv_path, header=None)
    # Sort the dataframe by frame number
    features_df = features_df.sort_values(by=1, key=lambda col: col.astype(float))
    # Get the frame numbers and last 35 columns
    frame_numbers = features_df.iloc[:, :2]
    # name the columns of the frame numbers dataframe
    frame_numbers.columns = ['seq', 'frame_num']
    # Add a column to keep track of whether to keep the frame
    frame_numbers['keep'] = True

    # drop column 0 and 5-34

    features_df = features_df.drop(features_df.columns[[0] + list(range(5, 35))], axis=1)

    
    # Initialize a buffer for valid data and frame tracking
    valid_data = []
    start_index = None

    # Iterate through rows in the DataFrame
    for index, row in features_df.iterrows():
        if row.isnull().values.any() or row.empty:
            # If the row contains blanks, process the current sequence
            if len(valid_data) >= 30:  # Check if the sequence has at least 30 frames
                last_frame = frame_numbers.iloc[index-1]['frame_num']
                start_frame = frame_numbers.iloc[start_index]['frame_num']
                tensor = torch.tensor(valid_data, dtype=torch.float32)
                tensor_path = os.path.join(lm_dir, f"{clip}_frames_{start_frame}_{last_frame}.pt")
                torch.save(tensor, tensor_path)  # Save the tensor
                # Add metadata for this tensor
                metadata_list.append([clip, start_frame, last_frame, tensor_path])
            else:
                # set 'keep' to false for the frames from start_frame to the current frame
                frame_numbers.iloc[start_index:index]['keep'] = False

            # In frame_numbers, set 'keep' to false for the current frame (since it has nan values)
            frame_numbers.iloc[index]['keep'] = False
                
            # Reset the valid data buffer
            valid_data = []
            start_index = None
        else:
            # Add the row to valid data
            if start_index is None:
                start_index = index  # Record the start frame
            valid_data.append(row.values)
            frame_numbers.iloc[index]['keep'] = True

    # Handle any remaining valid data after the loop
    if len(valid_data) >= 30:
        last_frame = frame_numbers.iloc[index-1]['frame_num']
        start_frame = frame_numbers.iloc[start_index]['frame_num']
        tensor = torch.tensor(valid_data, dtype=torch.float32)
        tensor_path = os.path.join(new_lm_dir, f"{clip}_frames_{start_frame}_{last_frame}.pt")
        torch.save(tensor, tensor_path)
        metadata_list.append([clip, start_frame, frame_numbers.iloc[-1], tensor_path])

    # Append the frame numbers to the keep_frames DataFrame
    keep_frames = pd.concat([keep_frames, frame_numbers], ignore_index=True)

# Save the metadata as a CSV
metadata_df = pd.DataFrame(metadata_list, columns=["Clip", "StartFrame", "EndFrame", "TensorPath"])
metadata_df.to_csv(os.path.join(new_lm_dir, "metadata.csv"), index=False)

# Save frame numbers as a CSV
keep_frames.to_csv(os.path.join(new_lm_dir, "keep_frames.csv"), index=False)

