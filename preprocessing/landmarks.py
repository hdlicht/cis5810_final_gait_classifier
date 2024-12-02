import os
import cv2
import mediapipe as mp
import pandas as pd

# Define path to directory containing frame subdirectories
root_dir = "gcs_images/gavd_frame_images"
dest_dir = "landmark_files"

# Traverse each subdirectory (each subdirectory is one video)
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"Processing video: {subdir}")
        all_landmarks = []  # Collect landmarks for all frames in this subdir
        
        # Reinitialize Mediapipe Pose for each video
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False)  # Set to False for video frames
        
        # Process each frame
        frame_files = sorted(os.listdir(subdir_path))
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(subdir_path, frame_file)
            image = cv2.imread(frame_path)

            if image is not None:
                # Process with Mediapipe
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks = [f"{lm.x},{lm.y},{lm.z}" for lm in results.pose_landmarks.landmark]
                    all_landmarks.append(landmarks)
            else:
                print(f"Warning: Could not read frame {frame_file}")

        # Save to CSV file
        if all_landmarks:
            csv_path = os.path.join(root_dir, f"{subdir}_landmarks.csv")
            df = pd.DataFrame(all_landmarks)
            df.to_csv(csv_path, index=False, header=False)
            print(f"Saved landmarks for video '{subdir}' to {csv_path}")
        else:
            print(f"No landmarks detected for video '{subdir}'.")

        # Clean up the Mediapipe Pose instance
        pose.close()  # Optional but can be good practice to free up resources
