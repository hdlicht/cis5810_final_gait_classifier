import cv2
import time
import os
from PIL import Image
import torch
from mp_landmarks import get_landmarks, start_mediapipe
from bbox import crop_square
import imageio

def save_gif(frames, output_dir='demo_app/frames', fps=30):
    # # Save the frames as a GIF
    # frames[0].save(f'{output_dir}/output.gif', save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
    # print("GIF saved successfully")
    with imageio.get_writer(os.path.join(output_dir,f"output.gif"), mode="I") as writer:
        for frame in frames:
            writer.append_data(frame)


def start_recording(output_dir='demo_app/frames', seconds=3, count_down = 5):
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
        while now < seconds:
            ret, frame = cap.read()
            # rotate frame 90 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
            cv2.putText(frame, f"Recording in {seconds-now}", (5, frame_height//2),
                        font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            yield frame
            # # Show the frame with countdown
            # cv2.imshow('Webcam Feed', frame)
            # cv2.waitKey(1)

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
            
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            now = time.time()-start_time
            mp_stuff = get_landmarks(landmark_detector, frame)
            visible = False
            bbox = [0, 0, 1, 1]

            if mp_stuff:
                marks, bbox, visible = mp_stuff
            if visible:
                square_frame = crop_square(frame, bbox)

                frames.append(Image.fromarray(square_frame))
                seq_marks.append(marks)
                if output:
                    cv2.imwrite(f'{output_dir}/{frame_count:03d}.jpg', square_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Recording for {seconds-now}", (5, frame_height//2),
                        font, 2, (0, 255, 0), 3, cv2.LINE_AA)
            # draw bounding box around the frame (red if not visible)
            if not visible:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # # Show the frame with countdown
            # cv2.imshow('Webcam Feed', frame)
            yield frame
            frame_count += 1
            # cv2.waitKey(1)

        if frames != []:
            seq_marks = torch.tensor(seq_marks)
            return frames, seq_marks
        return None, None

    # Show countdown before starting the recording
    for frame in countdown(count_down):
        yield frame
    frames, seq_marks = None, None
    recording = record(seconds)
    try:
        while True:
            frame = next(recording)
            yield frame
    except StopIteration as e:
        stuff = e.value
        if stuff is not None:
            frames, seq_marks = stuff
            

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

    if frames is not None:
        # downsample the frames by 2 and reduce to 30 frames
        frames = frames[::2]
        frames = frames[:30]
        # also downsample the landmarks
        seq_marks = seq_marks[::2]
        seq_marks = seq_marks[:30]
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

