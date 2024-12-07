# app.py

import gradio as gr
import torch
import numpy as np
from record import start_recording
from dino_features import init_dino_model, get_dino_features
from inferences import *
import logging as log
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

dino = None
cam_classifier = None
mega_model = None
dataset_clips = None
dataset_features = None
label_mapping = None
angle_mapping = None

prediction = None
confidence = None
features = None
good_data = False

# Your startup function
def load_models():
    global dino, cam_classifier, mega_model, dataset_features, dataset_clips, label_mapping, angle_mapping
    dino = init_dino_model()
    # mp_model = load_mp_model('weights/weights/final_big_mpmodel_20_3.pth')
    # dino_model = load_dino_model('weights/weights/final_dinomodel_20_3.pth')
    cam_classifier = load_cam_classifier('weights/final_cam_class.pth')
    mega_model = load_mega_model('weights/final_all_fft_lstm_128_2.pth')

    # Load the saved data
    loaded_data = torch.load("demo_app/feature_vectors/lstm_fft_averaged.pth")
    label_mapping = np.load("demo_app/label_mapping.npy",allow_pickle=True).item()
    angle_mapping = np.load("demo_app/angle_mapping.npy",allow_pickle=True).item()

    # Extract features and clip IDs
    dataset_features = loaded_data["average_vectors"]
    dataset_clips = list(loaded_data["clip_ids"])

def nearest_neighbors(new_features):
    # Calculate the nearest neighbors
    global dino, cam_classifier, mega_model, dataset_features, dataset_clips, label_mapping, angle_mapping
    new_features = new_features.unsqueeze(0)
    distances = torch.cdist(dataset_features, new_features)
    distances = distances.squeeze()
    _, indices = torch.topk(distances, 3, largest=False)
    labels = []
    angles = []
    gifs = []
    for i in indices:
        clip = dataset_clips[i]
        label = label_mapping.get(clip, None)
        angle = angle_mapping.get(clip, None)
        gif_path =  f"data/gavd_dataset/gifs/{label}/{angle}/{clip}.gif"
        labels.append(label)
        angles.append(angle)
        gifs.append(gif_path)
    return labels, angles, gifs

def recording_button():
    global dino, cam_classifier, mega_model, prediction, confidence, features
    frames, landmarks = None, None
    recording = start_recording(seconds=4, count_down=5)
    try:
        while True:
            frame = next(recording)
            yield gr.update(value=frame, visible=True), "Recording...", None
    except StopIteration as e:
        stuff = e.value
        if stuff is not None:
            frames, landmarks = stuff
    
    if frames is None:
        print("No frames recorded")
        yield gr.update(visible=False), "No frames recorded", None
    elif landmarks is None:
        print("Bad data. Try again.")
        yield gr.update(visible=False), f"Bad data. Try again.", None
    else:
        # change image to output gif

        print(f"Recorded {len(frames)} frames")
        yield gr.update(value="demo_app/frames/output.gif"), f"Recorded {len(frames)} frames", None

    if frames is not None and landmarks is not None:
        log.info("Video processing completed. Extracting features...")
        features = get_dino_features(dino, frames)
        cam_view = predict_cam_view(cam_classifier, frames[:10])
        print(f'Cam View: {cam_view}')    
        result = f'Cam View: {cam_view}'

        yield gr.update(value="demo_app/frames/output.gif"), result, "See nearest neighbors"
        
        # Make predictions
        log.info("Making predictions...")
        prediction, confidence, features = predict_mega_model(mega_model, landmarks, features, cam_view)
        result += f'\nPredicted Class: {prediction}, Confidence: {confidence:.2f}'
        good_data = True
        yield gr.update(value="demo_app/frames/output.gif"), result, gr.update(visible=True)


    else:
        result = "No frames recorded"
    print("done")
    return None, result

def on_features_ready():
    global features, good_data
    labels, angles, gif_paths  = nearest_neighbors(features)

    # Update GIFs and labels in the UI
    return gr.update(value=gif_paths[0], visible=True, label=f"{labels[0]} - {angles[0]}"), \
        gr.update(value=gif_paths[1], visible=True, label=f"{labels[1]} - {angles[1]}"), \
        gr.update(value=gif_paths[2], visible=True, label=f"{labels[2]} - {angles[2]}")

# Create the Gradio interface
with gr.Blocks() as demo:
    load_btn = gr.Button("Load Models")
    btn = gr.Button("Start Recording")
    output_video = gr.Image()
    output_text = gr.Textbox()
    btn2 = gr.Button("See nearest neighbors", visible=False)
    # Grid for GIF outputs with captions
    with gr.Row():
        gif1 = gr.Image(visible=False)  # GIF placeholder 1
        gif2 = gr.Image(visible=False)  # GIF placeholder 2
        gif3 = gr.Image(visible=False)  # GIF placeholder 3

    load_btn.click(fn=load_models, inputs=None, outputs=None)
    btn.click(fn=recording_button, inputs=None, outputs=[output_video, output_text, btn2])
    btn2.click(fn=on_features_ready, inputs=None, outputs=[gif1, gif2, gif3])
# Launch the app
demo.launch()
