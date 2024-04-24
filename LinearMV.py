import os
import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, interact_manual, FloatSlider, IntSlider
import math

from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np

# local modules
from util import Timer, Event, normalize_image, animate, load_events,load_events_volt, plot_3d, event_slice

def kalman_filter(event_data, process_noise=0.1, observation_noise=0.5):
    print('Filtering with Kalman Filter, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
    frame_timestamps = frame_timestamps
    events_per_frame = 2e4
    with Timer('Filtering'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        frame_idx = 0
        max_frame_idx = len(frames) - 1
        #TODO define MV filter parameter like x(list), F0，xxx, 
        #you may predefine diffenet Fk,Rk ... to save time

        for i, e in enumerate(events):
            if frame_idx < max_frame_idx:
                # print(e.t)

                if e.t >= frame_timestamps[frame_idx + 1]:
                    frame_idx += 1

                    # Process image_state and save to folder
                    processed_image_state = 2 ** image_state - 1
                    # processed_image_state = image_state
                    save_image(processed_image_state, frame_idx+2, folder_path="D:/2024/3DGS/PureEventFilter/data/mic_colmap_easy/output_images")

            # Kalman filter update step
            # You need to implement the Kalman filter equations here
            
            # Prediction step: x_k = F * x + w, P_k = F * P * F^T + Q
            # Update step: K = P_k * H^T * (H * P_k * H^T + R)^-1
            #              x = x_k + K * (z - H * x_k), P = (I - K * H) * P_k

            # Here, 'e' represents the event data point
            
            # Update time_surface and image_state
            
            # Keep track of image state and time surface
            time_surface[e.y, e.x] = e.t

            #update log-intensity through 
            image_state[e.y, e.x] 

            if i % events_per_frame == 0:
                # Perform Kalman filter update step every events_per_frame events
                pass  # You need to add your Kalman filter update step here

                # Append the filtered image to the image list
                image_list.append(np.copy(image_state))
    return animate(image_list, 'Kalman Filter')
def save_image(image, index, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{index:05d}.png")
    # Saving image
    # print("output")
    # Convert to uint8 before saving with OpenCV
    # image_uint8 = (image * 255).astype(np.uint8)
    # # Saving image with OpenCV
    # cv2.imwrite(file_path, image_uint8)
    plt.imsave(file_path, image, cmap='gray')