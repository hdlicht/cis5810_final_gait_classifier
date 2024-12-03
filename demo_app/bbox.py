import cv2
import numpy as np
        
def crop_square(frame, bbox):
    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape
    # Crop the frame to the bounding box
    x_min, y_min, x_max, y_max = bbox
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
    resized_frame = cv2.resize(square_crop, (224, 224))
    return resized_frame

