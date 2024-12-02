import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

def get_all_landmarks(frames):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    sequence_landmarks = []
    for frame in frames:
        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(rgb_frame)

        # Extract landmarks if detected
        if results.pose_landmarks:
            # sequence_landmarks.append([(lm.x, lm.y, lm.z) for lm in landmarks])
            points = [[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] for landmark in results.pose_landmarks.landmark]
            bbox = [min([p[0] for p in points]), min([p[1] for p in points])-10, max([p[0] for p in points]), max([p[1] for p in points])]
            visibility = [lm.visibility for lm in results.pose_landmarks.landmark]
            visible = np.all(np.array(visibility) > 0.6)
            marks = [f"{lm.x},{lm.y},{lm.z}" for lm in results.pose_landmarks.landmark]
            sequence_landmarks.append(marks)

        else:
            sequence_landmarks.append(None)
    return sequence_landmarks

def get_landmarks(frame):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_frame)

    # Extract landmarks if detected
    if results.pose_landmarks:
        # sequence_landmarks.append([(lm.x, lm.y, lm.z) for lm in landmarks])
        points = [[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] for landmark in results.pose_landmarks.landmark]
        bbox = [min([p[0] for p in points]), min([p[1] for p in points])-10, max([p[0] for p in points]), max([p[1] for p in points])]
        visibility = [lm.visibility for lm in results.pose_landmarks.landmark]
        # visible if landmarks 24-32 are visible
        visible = np.all(np.array(visibility[24:32]) > 0.6)
        marks = [[lm.x,lm.y,lm.z] for lm in results.pose_landmarks.landmark]
        return marks, bbox, visible
