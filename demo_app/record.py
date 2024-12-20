import cv2
import time
import numpy as np
import os
from PIL import Image
import torch
from mp_landmarks import get_landmarks, start_mediapipe
from bbox import crop_square
import imageio

def save_gif(frames, output_dir='demo_app/frames'):
    # # Save the frames as a GIF
    frames = frames[::3]
    with imageio.get_writer(os.path.join(output_dir,f"output.gif"), mode="I",loop=0) as writer:
        for frame in frames:
            # Covert from BGR to RGB format

            frame_np = np.array(frame)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_np)


def start_recording(output_dir='demo_app/frames', seconds=5, count_down = 3):
    # Initialize the webcam
    cap = cv2.VideoCapture(4)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    # Set video frame width and height (optional)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmark_detector = start_mediapipe()

    # Countdown function to show the countdown on the screen
    def countdown(seconds):
    # Start recording after countdown
        start_time = time.time()
        now = time.time()-start_time
        ready = False
        while not ready:
            ret, frame = cap.read()
            # rotate frame 90 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                print("Error: Failed to grab frame")
                break
            visible = False
            bbox = [0, 0, 1, 1]
            now = time.time()-start_time
            mp_stuff = get_landmarks(landmark_detector, frame)
            if mp_stuff:
                marks, bbox, visible = mp_stuff
            if visible:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            # Add countdown text to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Get in the frame!", (5, frame_height//2), #mp_conf:.2f
                        font, 1, (0, 0, 255), 3, cv2.LINE_AA)
            yield frame
            if visible:
                ready = True
        start_time = time.time()
        now = time.time()-start_time
        while now < seconds:
            ret, frame = cap.read()
            # rotate frame 90 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                print("Error: Failed to grab frame")
                break
            visible = False
            bbox = [0, 0, 1, 1]
            now = time.time()-start_time
            mp_stuff = get_landmarks(landmark_detector, frame)
            if mp_stuff:
                marks, bbox, visible = mp_stuff
            if visible:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            # Add countdown text to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Start in {seconds-now:.2f}", (5, frame_height//2), #mp_conf:.2f
                        font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            yield frame
            # # Show the frame with countdown
            # cv2.imshow('Webcam Feed', frame)
            # cv2.waitKey(1)

    def record(seconds, output=True):
        # Frame capture rate
        frame_interval = 0.01  # Time between frames in seconds
        
        start_time = time.time()
        now = time.time() - start_time
        frame_count = 0
        frames = []

        while now < seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            # Rotate the frame
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add frame to the list
            frames.append(Image.fromarray(frame))
            if output:
                cv2.imwrite(f'{output_dir}/{frame_count:03d}.jpg', frame)

            # Display recording text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Recording", (5, frame_height // 2),
                        font, 2, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Show the frame
            yield frame
            frame_count += 1

            # Sleep to maintain 30 FPS
            time.sleep(frame_interval)
            now = time.time() - start_time
        print(f"Recorded {frame_count} frames")
        return frames

    # Show countdown before starting the recording
    for frame in countdown(count_down):
        yield frame
    frames = None
    recording = record(seconds)
    try:
        while True:
            frame = next(recording)
            yield frame
    except StopIteration as e:
        frames = e.value
    
    if frames:
        cropped_frames = []
        seq_marks = []
        for frame in frames:
            frame_np = np.array(frame)
            mp_stuff = get_landmarks(landmark_detector, frame_np)
            if mp_stuff:
                marks, bbox, visible = mp_stuff
                if not visible:
                    print("Bad data. Try again.")
                    return None, None
                
                seq_marks.append(marks)

        seq_marks = torch.tensor(seq_marks) if seq_marks else None
            

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

    if frames is not None:
        # Save the frames as a GIF
        save_gif(frames, output_dir)

    return frames, seq_marks

if __name__ == "__main__":
    frames, seq_marks = start_recording(output_dir='demo_app/frames', seconds=5, count_down=3)
    if frames.size(0) > 0:
        print("Recording completed")
        print("Frames saved in 'demo_app/frames' directory")
        torch.save(seq_marks, 'demo_app/frames/landmarks.pt')
        print(seq_marks.shape)
    else:
        print("Recording failed")

