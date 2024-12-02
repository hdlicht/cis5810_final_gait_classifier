import os
import pandas as pd
import numpy as np
import pickle
import cv2
import argparse
import mediapipe as mp


def extract_and_crop_frames(seq, video_id, frame_num, video_info, bbox, output_dir):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # Open the video file
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

        video_path = f'./videos/videos/{video_id}.mp4'
        cap = cv2.VideoCapture(video_path)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        mp_selfie_segmentation = mp.solutions.selfie_segmentation


        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if the frame number is within the total frame count
        if frame_num >= total_frames:
            print(f"Frame {frame_num} exceeds total frames in the video: {total_frames}")
            return None

        # Set the video to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # print(frame.shape)
            # cv2.imshow("og frame",frame)
            # # Wait indefinitely until a key is pressed
            # cv2.waitKey(0)

            # Optionally, close the window afterward
            cv2.destroyAllWindows()

                # Perform segmentation
            result = selfie_segmentation.process(frame)

            # Generate a mask where the person is segmented
            mask = result.segmentation_mask > 0.1  # Threshold for a cleaner mask
            print(mask.shape)
            # Find the bounding box around the segmented person
            coords = np.argwhere(mask)
            print(coords)
            if coords.size > 2:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # Crop the frame to the bounding box
                width = x_max - x_min
                height = y_max - y_min 
                # Determine the longer side (to make the square crop)
                square_side = max(width, height)

                # Calculate new top and left to center the crop
                center_x = x_min + width // 2
                center_y = y_min + height // 2
                new_left = max(0, center_x - square_side // 2)
                new_top = max(0, center_y - square_side // 2)

                # Ensure the crop stays within the image boundaries
                new_right = int(min(new_left + square_side, frame_width - 1))
                new_bottom = int(min(new_top + square_side, frame_height - 1))

                # Crop the image based on the new square dimensions
                square_crop = frame[new_top:new_bottom, new_left:new_right]

                # Resize the square crop to 224x224
                print(square_crop.shape)
                resized_frame = cv2.resize(square_crop, (224, 224))

                # Save the resized, cropped frame
                output_dir = os.path.join(output_dir, f"{seq}")
                output_file = os.path.join(output_dir, f"{seq}_frame_{frame_num}.jpg")

                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(output_file, resized_frame)
                #print(f"Saved resized cropped frame: {output_file}")
        else:
            print(f"Failed to read frame {frame_num}")

        # Release the video capture
        cap.release()

def extract_frames(filename):

    data_path = 'videos'
    output_path = 'frames5'
    need_extract = pd.read_pickle(filename)

    for idx, row in need_extract.iterrows():
        if idx % 10 == 0:
            print(f"Processing row {idx}")
        video_id = row['id']
        seq = row['seq']
        video_path = f'./videos/{video_id}.mp4'
        video_info = row['vid_info']
        bbox = row['bbox']  # Assuming bbox is stored as a string, so convert it to tuple
        frame_num = row['frame_num']
        extract_and_crop_frames(seq, video_id, frame_num, video_info, bbox, output_path)
        need_extract.at[idx, 'frame_extracted'] = True

    pd.to_pickle(need_extract, filename)

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="A script that accepts arguments")
    
    # Add argument (you can add more if needed)
    parser.add_argument('arg_value', type=str, help="Argument passed to the script")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the argument value
    extract_frames(args.arg_value)