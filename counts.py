import os
import pandas as pd
import torch

# Directory containing landmark CSV files
lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/landmarks"
dino_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/dino_features"

new_lm_dir = "/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/all_features"
# make the new directory if it does not exist
os.makedirs(new_lm_dir, exist_ok=True)

# Initialize lists for tensors and metadata
# tensor_list = []
# metadata_list = []
# keep_frames = pd.DataFrame()
# keep_frames['seq'] = []
# keep_frames['frame_num'] = []
# keep_frames['keep'] = []

# iterate through the CSV files
for file in [x for x in os.listdir(lm_dir) if x.endswith('.csv')]:
    # Get the clip name
    clip = file.split('_')[0]
    # Get the full CSV path
    csv_path = os.path.join(lm_dir, file)
    # Read the CSV
    features_df = pd.read_csv(csv_path, header=None)
    features_df = features_df.drop(features_df.columns[[0] + list(range(5, 35))], axis=1)
    len_df = len(features_df)
    # open the dino feature file for the clip
    dino_path = os.path.join(dino_dir, f"{clip}_features.pt")
    dino_features = torch.load(dino_path)
    len_dino = dino_features.shape[0]
    # assert that the lengths are the same
    assert len_df == len_dino, f"Lengths do not match for {clip}: {len_df} vs {len_dino}"
    # add dino features to the dataframe
    features_df = pd.concat([features_df, pd.DataFrame(dino_features.numpy())], axis=1)

    # Assuming 'features_df' is your DataFrame
    num_landmark_columns = 69  # Number of landmark columns
    num_dino_features = 768  # Number of DINO feature columns

    # Create a list of column names
    columns = (
        ['frame_num'] + 
        [f'landmark_{i}' for i in range(1, num_landmark_columns + 1)] + 
        [f'dino_feature_{i}' for i in range(1, num_dino_features + 1)]
    )
    features_df.columns = columns


    # Initialize a buffer for valid data and frame tracking
    valid_data = []
    start_index = None

    # Iterate through rows in the DataFrame
    for index, row in features_df.iterrows():
        if row.isnull().values.any() or row.empty:

            # If the row contains blanks, process the current sequence
            if len(valid_data) >= 30:  # Check if the sequence has at least 30 frames
                last_frame = features_df.iloc[index - 1]['frame_num']
                start_frame = features_df.iloc[start_index]['frame_num']
                tensor = torch.tensor(valid_data, dtype=torch.float32)
                tensor_path = os.path.join(lm_dir, f"{clip}_frames_{int(start_frame)}_{int(last_frame)}.pt")
                torch.save(tensor, tensor_path)  # Save the tensor
                
            # Reset the valid data buffer
            valid_data = []
            start_index = None
        else:
            # Add the row to valid data
            if start_index is None:
                start_index = index  # Record the start frame
            valid_data.append(row.values)

    # Handle any remaining valid data after the loop
    if len(valid_data) >= 30:
        last_frame = features_df.iloc[index-1]['frame_num']
        start_frame = features_df.iloc[start_index]['frame_num']
        tensor = torch.tensor(valid_data, dtype=torch.float32)
        tensor_path = os.path.join(new_lm_dir, f"{clip}_frames_{int(start_frame)}_{int(last_frame)}.pt")
        torch.save(tensor, tensor_path)



