import os 
import cv2
import numpy as np
import imageio
import pandas as pd

image_path = '/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/frame_images'
save_dir = f'/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/gifs'
annotations = '/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/annotations/gavd_annotations.pkl'

# Load the annotations
df = pd.read_pickle(annotations)
# get the unique values of seq, cam_view, gait_pat
df = df.drop_duplicates(subset=['seq', 'cam_view', 'gait_pat'])

for index, row in df.iterrows():
    seq = row['seq']
    cam_view = row['cam_view']
    gait_pat = row['gait_pat']
    image_dir = os.path.join(image_path, seq)
    if not os.path.exists(image_dir):
        print(f'{image_dir} does not exist')
        continue
    # extract the frame number from each file in the directory
    frames = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(image_dir)]
    frames = np.sort(frames)

    # downsample the frames
    downsampled_frames = frames[::3]
    # only take the first 40 frames
    if len(downsampled_frames) > 40:
        downsampled_frames = downsampled_frames[:40]
    print(f'frames: {downsampled_frames}')

    # save the downsampled images
    save_dir = f'/home/henry/robo/cis5810/final/cis5810_final_gait_classifier/data/gavd_dataset/gifs/{gait_pat}/{cam_view}'
    os.makedirs(save_dir, exist_ok=True)
    # Save the frames as a GIF that loops
    gif_path = os.path.join(save_dir, f"{seq}.gif")
    
    # Create the GIF
    with imageio.get_writer(gif_path, mode="I", loop=0) as writer:  # loop=0 for infinite looping
        for frame in downsampled_frames:
            file = os.path.join(image_dir, f'{seq}_frame_{frame}.jpg')
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)

    

