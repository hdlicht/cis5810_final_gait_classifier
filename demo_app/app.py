# app.py

import gradio as gr
import torch
from record import start_recording
from dino_features import init_dino_model, get_dino_features
from inferences import load_dino_model, load_mp_model, predict
import logging as log

def gradio_recording_button():
    dino = init_dino_model()
    mp_model = load_mp_model('weights/weights/final_big_mpmodel_20_3.pth')
    dino_model = load_dino_model('weights/weights/final_dinomodel_20_3.pth')
    frames, landmarks = None, None
    recording = start_recording()
    try:
        while True:
            frame = next(recording)
            yield frame, "Recording..."
    except StopIteration as e:
        stuff = e.value
        if stuff is not None:
            frames, landmarks = stuff
    if frames is not None:
        yield None, f"Recorded {len(frames)} frames"
    else:
        yield None, "No frames recorded"

    # Process the final data after streaming
    if frames is not None and landmarks is not None:
        log.info("Video processing completed. Extracting features...")
        features = get_dino_features(dino, frames)
        
        # Make predictions
        log.info("Making predictions...")
        mp_result, mp_conf = predict(mp_model, landmarks)
        dino_result, dino_conf = predict(dino_model, features)
        
        result = f"MediaPipe: {mp_result} ({mp_conf:.2f})\nDINO: {dino_result} ({dino_conf:.2f})"
        log.info("Prediction completed.")
        return None, result

# Run the load model function in the background when the app starts
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dino = init_dino_model()
# mp_model = load_mp_model('weights/weights/final_big_mpmodel_20_3.pth')
# dino_model = load_dino_model('weights/weights/final_dinomodel_20_3.pth')

# Create the Gradio interface
with gr.Blocks() as demo:
    btn = gr.Button("Start Processing")
    # output_text = gr.Textbox()
    output_video = gr.Image()
    output_text = gr.Textbox()

    btn.click(fn=gradio_recording_button, inputs=None, outputs=[output_video, output_text])


# Launch the app
demo.launch()