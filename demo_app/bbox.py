import cv2
import mediapipe as mp
import numpy as np

def get_bbox(frame):

    # Initialize MediaPipe Selfie Segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # Set up the segmentation model
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation
        result = selfie_segmentation.process(rgb_frame)

        # Generate a mask where the person is segmented
        mask = result.segmentation_mask > 0.1  # Threshold for a cleaner mask

        # Find the bounding box around the segmented person
        coords = np.argwhere(mask)
        if coords.size > 2:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            return x_min, y_min, x_max, y_max
        else:
            return None
        
def crop_square(frame):
    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape
    # Crop the frame to the bounding box
    x_min, y_min, x_max, y_max = get_bbox(frame)
    width = x_max - x_min
    height = y_max - y_min 
    # Determine the longer side (to make the square crop)
    square_side = max(width, height)+10

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
    return resized_frame

