#TODO 1: use linear interpolation based on time for event pixel, instead of the last change
# this can be alleviate by 3DGSEventLoss, use relu to get a wider bound
# 2: better noise 
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

def initialState(event_data,c=0.1):
    # Initialize an array of the same size as diff_state to store the event intensity corresponding to the minimum timestamp
    events, height, width = event_data.event_list, event_data.height, event_data.width
    max_count =  1.0*height * width
    init_state = np.zeros((height, width), dtype=np.float32)
    # Traverse all events
    i = 0
    for event in events:
        # Get the position and event intensity of the event
        x, y, p = event.x, event.y, event.p
        
        # If the current position is not initialized or the event intensity of the current event is greater than the recorded event intensity, update the event intensity
        if init_state[x, y] == 0:
            init_state[x, y] = p*c
    
        # Check if all positions have been initialized
        # The probability of initializing all positions is very low, so it's fine to assign 0 to the background
        # Therefore, only take the first max_count events to initialize
        i = i + 1
        if i > max_count:
            break
    return init_state

def kalman_filter(event_data,c = 0.1):
    print('Filtering with Kalman Filter, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
    frame_timestamps = frame_timestamps
    events_per_frame = 2e4
    with Timer('Filtering'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        # diff_state = np.zeros((height, width), dtype=np.float32)
        diff_state = initialState(event_data,c)
        # diff_state = np.ones((height, width), dtype=np.float32)

        # covariance_state = np.full((height, width),c*c, dtype=np.float32)
        covariance_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        frame_idx = 0
        max_frame_idx = len(frames) - 1
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

            #parameter calculate
            Fk = 1
            # #Sk is wkvk, Qk is wkwk, R is vkvk, normally Qk> RK
            # Sk=0
            # Qk=0.001 
            # Rk=0.05
            Sk=0
            Qk=0.1
            Rk=1
            #update
            P = covariance_state[e.y, e.x]
            K = (Fk*P*1/c + Sk)/(1/c*P*1/c + Rk)
            diff_state[e.y, e.x] = Fk*diff_state[e.y, e.x] + K*(e.p - 1/c * diff_state[e.y, e.x])
            covariance_state[e.y, e.x] = (Fk-K*1/c)*P*(Fk-K*1/c)+Qk+K*Rk*K-2*Sk*K           
            # Keep track of image state and time surface
            time_surface[e.y, e.x] = e.t

            #update log-intensity through 
            image_state[e.y, e.x] = image_state[e.y, e.x] + diff_state[e.y, e.x]

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