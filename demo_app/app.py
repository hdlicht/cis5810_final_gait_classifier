# app.py

import gradio as gr
from record import start_recording
from bbox import crop_square
from mp_landmarks import get_landmarks
from dino_features import get_dino_features
from inferences import load_dino_model, load_mp_model, predict


def gradio_recording_button():
    # Call the start_recording function and get the result
    frames = start_recording(output_dir='frames', seconds=3, count_down=5)
    landmarks = get_landmarks(frames)
    features = get_dino_features(frames)
    # Load the model
    mp_model = load_mp_model('weights/final_big_mpmodel_20_3.pth')
    dino_model = load_dino_model('weights/final_dinomodel_20_3.pth')
    # Make predictions
    mp_result, mp_conf = predict(mp_model, landmarks)
    dino_result, dino_conf = predict(dino_model, features)
    result = f"MediaPipe: {mp_result} ({mp_conf:.2f})\nDINO: {dino_result} ({dino_conf:.2f})"
    return result

# Create the Gradio interface
with gr.Blocks() as demo:
    # Create a button to start recording
    gr.Interface(fn=gradio_recording_button, 
                inputs=None,
                outputs="text",
                title="Start Recording")
    # change the generate button text to "Start Recording"


# Launch the app
demo.launch()
