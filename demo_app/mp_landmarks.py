import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


def start_mediapipe():
    mp_pose = mp.solutions.pose
    detector = mp_pose.Pose(static_image_mode=False)
    return detector

def get_landmarks(detector, frame):
    # Initialize MediaPipe Pose model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = detector.process(rgb_frame)

    # Extract landmarks if detected

    if results.pose_landmarks:
        # sequence_landmarks.append([(lm.x, lm.y, lm.z) for lm in landmarks])
        points = [[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] for landmark in results.pose_landmarks.landmark]
        bbox = [min([p[0] for p in points]), min([p[1] for p in points])-10, max([p[0] for p in points]), max([p[1] for p in points])]
        visibility = [lm.visibility for lm in results.pose_landmarks.landmark]
        # visible if landmarks 27-32 are visible
        visible = np.average(np.array(visibility[27:33])) > 0.5
        relevant_landmarks = [results.pose_landmarks.landmark[0]] + results.pose_landmarks.landmark[11:]

        marks = [value for lm in relevant_landmarks for value in (lm.x, lm.y, lm.z)]
        
        return marks, bbox, visible
