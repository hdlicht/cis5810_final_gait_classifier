import gradio as gr
import torch
from record import start_recording
from dino_features import init_dino_model, get_dino_features
from inferences import load_dino_model, load_mp_model, predict
import logging as log

def gradio_recording_button():
    # Load models
    dino = init_dino_model()
    mp_model = load_mp_model('weights/weights/final_big_mpmodel_20_3.pth')
    dino_model = load_dino_model('weights/weights/final_dinomodel_20_3.pth')

    frames, landmarks = None, None
    recording = start_recording(seconds=5)

    # Start video recording
    try:
        while True:
            frame = next(recording)
            # Display the current frame while recording
            yield gr.update(value=frame, visible=True), "Recording..."
    except StopIteration as e:
        # Finalize recording
        stuff = e.value
        if stuff is not None:
            frames, landmarks = stuff

    # Hide the video display after recording ends
    yield gr.update(visible=False), f"Recorded {len(frames)} frames" if frames else "No frames recorded"

    # Process the final data after recording
    if frames is not None and landmarks is not None:
        log.info("Video processing completed. Extracting features...")
        features = get_dino_features(dino, frames)

        # Make predictions
        log.info("Making predictions...")
        mp_result, mp_conf = predict(mp_model, landmarks)
        dino_result, dino_conf = predict(dino_model, features)

        result = f"MediaPipe: {mp_result} ({mp_conf:.2f})\nDINO: {dino_result} ({dino_conf:.2f})"
        log.info("Prediction completed.")
        yield None, result

# Create the Gradio interface
with gr.Blocks() as demo:
    btn = gr.Button("Start Processing")
    output_video = gr.Image(visible=False)  # Initially hidden
    output_text = gr.Textbox()

    btn.click(
        fn=gradio_recording_button, 
        inputs=None, 
        outputs=[output_video, output_text]
    )

# Launch the app
demo.launch()
