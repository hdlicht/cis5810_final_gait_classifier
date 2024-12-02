import cv2
import time
import os
import torch
from mp_landmarks import get_landmarks


# Initialize the webcam
cap = cv2.VideoCapture(0)


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


def start_recording(output_dir='demo_app/frames', seconds=3, count_down = 5):

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    # Set video frame width and height (optional)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Countdown function to show the countdown on the screen
    def countdown(seconds):
    # Start recording after countdown
        start_time = time.time()
        now = time.time()-start_time
        while now < seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            now = time.time()-start_time
            
            # Add countdown text to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Recording in {seconds-now}", (5, frame_height//2),
                        font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Show the frame with countdown
            cv2.imshow('Webcam Feed', frame)
            cv2.waitKey(1)

    # Countdown function to show the countdown on the screen
    def record(seconds, output=True):
        # Start recording after countdown
        start_time = time.time()
        now = time.time()-start_time
        frame_count = 0
        frames = []
        seq_marks = []

        while now < seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            now = time.time()-start_time
            marks, bbox, visible = get_landmarks(frame)
            if visible:
                square_frame = crop_square(frame, bbox)
                frames.append(square_frame)
                seq_marks.append(marks)
                if output:
                    cv2.imwrite(f'{output_dir}/{frame_count:03d}.jpg', frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Recording for {seconds-now}", (5, frame_height//2),
                        font, 2, (0, 255, 0), 3, cv2.LINE_AA)
            # draw bounding box around the frame (red if not visible)
            if not visible:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Show the frame with countdown
            cv2.imshow('Webcam Feed', frame)
            frame_count += 1
            cv2.waitKey(1)
        frames = torch.stack(frames)
        seq_marks = torch.tensor(seq_marks)
        return frames, seq_marks

    # Show countdown before starting the recording
    countdown(count_down)
    frames = record(seconds)

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

    return frames

if __name__ == "__main__":
    frames = start_recording(output_dir='demo_app/frames', seconds=5, count_down=5)
    print(frames.shape)
    print("Recording completed")
    print("Frames saved in 'demo_app/frames' directory")

