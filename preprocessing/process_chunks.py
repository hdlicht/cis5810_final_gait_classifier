import cv2
import mediapipe as mp
import pandas as pd
import sys
import io
import logging
from PIL import Image
import numpy as np
from google.cloud import storage

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_image_from_gcs(image_path):
    bucket_name = 'gavd_frame_images'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(image_path)
    img_bytes = blob.download_as_bytes()
    img = Image.open(io.BytesIO(img_bytes))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose

# Function to process a chunk of data
def process_chunk(input_file, output_file):
    # Define columns for the landmarks DataFrame
    columns = ['seq', 'frame']
    columns.extend([f'landmark_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z']])
    landmarks_df = pd.DataFrame(columns=columns)

    # Load the data chunk
    data_chunk = pd.read_pickle(input_file)

    # Create a MediaPipe `Pose` object
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        # Iterate through each row in the chunk
        total = data_chunk.shape[0]
        for idx, row in data_chunk.iterrows():
            seq = row['seq']
            frame_num = row['frame_num']
            lm_row = [seq, frame_num]

            if frame_num % 2 == 0:

                # Construct the image path
                image_path = f'{seq}/{seq}_frame_{frame_num}.jpg'

                # Load the image
                try:
                    image = get_image_from_gcs(image_path)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    lm_row.extend([None] * 99)  # Fill with None if image is not found
                    landmarks_df.loc[idx] = lm_row
                    continue

                if image is not None:
                    # Convert the image to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process the image to detect landmarks
                    result = pose.process(image_rgb)

                    # Extract landmarks if detected
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks.landmark
                        for lm in landmarks:
                            lm_row.extend([lm.x, lm.y, lm.z])
                    else:
                        lm_row.extend([None] * 99)  # Fill with None if no landmarks detected

                    # Add the row to the DataFrame
                    landmarks_df.loc[idx] = lm_row
                else:
                    logger.warning(f"Could not load image {image_path}")
                    lm_row.extend([None] * 99)  # Fill with None if image is not found
                    landmarks_df.loc[idx] = lm_row
            
            if idx % (total // 100) == 0:
                logger.info(f"Processed {idx} of {total} images")

    # Save the landmarks DataFrame to CSV
    landmarks_df.to_pickle(output_file)
    logger.info(f"Saved landmarks to {output_file}")

if __name__ == "__main__":
    # Extract the chunk index from the input argument
    chunk_index = sys.argv[1]

    # Define input and output file paths based on the index
    input_file = f'data_chunk_{chunk_index}.pkl'
    output_file = f'landmarks_{chunk_index}.pkl'

    # Process the chunk and save landmarks to output CSV
    process_chunk(input_file, output_file)
