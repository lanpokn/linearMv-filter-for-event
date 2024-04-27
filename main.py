from ipywidgets import interact, fixed, interact_manual, FloatSlider, IntSlider
import math

from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np

# local modules
from util import Timer, Event, normalize_image, animate, load_events,load_events_volt, plot_3d, event_slice

from LinearMV import KalmanFilter,save_image

def high_pass_filter(event_data, cutoff_frequency=5):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    events_per_frame = 2e4
    with Timer('Reconstruction'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        for i, e in enumerate(events):
            beta = math.exp(-cutoff_frequency * (e.t - time_surface[e.y, e.x]))
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] + e.p
            time_surface[e.y, e.x] = e.t
            if i % events_per_frame == 0:
                beta = np.exp(-cutoff_frequency * (e.t - time_surface))
                image_state *= beta
                time_surface.fill(e.t)
                image_list.append(np.copy(image_state))
    return animate(image_list, 'High Pass Filter')
import os
import cv2
import matplotlib.pyplot as plt
def complementary_filter(event_data, cutoff_frequency=5.0):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
    frame_timestamps = frame_timestamps
    events_per_frame = 2e4
    with Timer('Reconstruction'):
        ##core algorithm begin
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        frame_idx = 0
        max_frame_idx = len(frames) - 1
        log_frame = np.full((height, width), 0.73, dtype=np.float32)
        # print(frame_timestamps[frame_idx + 1])
        for i, e in enumerate(events):
            if frame_idx < max_frame_idx:
                # print(e.t)

                if e.t >= frame_timestamps[frame_idx + 1]:
                    log_frame = np.copy(image_state)  # Update log_frame with the latest image_state
                    frame_idx += 1

                    # Process image_state and save to folder
                    processed_image_state = 2 ** image_state - 1
                    # processed_image_state = image_state
                    save_image(processed_image_state, frame_idx+2)

            beta = math.exp(-cutoff_frequency * (e.t - time_surface[e.y, e.x]))
            # image_state[e.y, e.x] = beta * image_state[e.y, e.x] \
            #                         + (1 - beta) * log_frame[e.y, e.x] + 0.1 * e.p
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] \
                        + (1 - beta) * 0 + 0.01 * e.p

            time_surface[e.y, e.x] = e.t
            if i % events_per_frame == 0:
                beta = np.exp(-cutoff_frequency * (e.t - time_surface))
                image_state = beta * image_state + (1 - beta) * (2 ** 0 - 1)
                # image_state = beta * image_state + (1 - beta) *  log_frame
                time_surface.fill(e.t)
                image_list.append(np.copy(image_state))
    return animate(image_list, 'Complementary Filter')
def leaky_integrator(event_data, beta=1.0,c=0.01):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
    # events_per_frame = 2e4
    frame_idx = 0
    max_frame_idx = len(frames) - 1
    with Timer('Reconstruction (simple)'):
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        for i, e in enumerate(events):
            if frame_idx < max_frame_idx:
                # print(e.t)

                if e.t >= frame_timestamps[frame_idx + 1]:
                    # log_frame = np.copy(image_state)  # Update log_frame with the latest image_state
                    frame_idx += 1

                    # Process image_state and save to folder
                    processed_image_state = 2 ** image_state - 1
                    # processed_image_state = image_state
                    save_image(processed_image_state, frame_idx+2)
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] + c*e.p
    # fig_title = 'Direct Integration' if beta == 1 else 'Leaky Integrator'
    return
# def save_image(image, index, folder_path="D:/2024/3DGS/PureEventFilter/data/boxes_6dof/output_images"):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     file_path = os.path.join(folder_path, f"{index:05d}.png")
#     # Saving image
#     # print("output")
#     # Convert to uint8 before saving with OpenCV
#     # image_uint8 = (image * 255).astype(np.uint8)-
#     # # Saving image with OpenCV
#     # cv2.imwrite(file_path, image_uint8)
#     plt.imsave(file_path, image, cmap='gray')

with Timer('Loading'):
    n_events = 1e8
    path_to_events = "D:/2024/3DGS/PureEventFilter/data/mic_colmap_easy/mic_volt.txt"
    event_data = load_events_volt(path_to_events, n_events)
    # path_to_events = "D:/2024/3DGS/PureEventFilter/data/boxes_6dof/events.zip"
    # event_data = load_events(path_to_events, n_events)           

event_data.add_frame_data('data/mic_colmap_easy')
# event_data.add_frame_data('data/boxes_6dof')

# complementary_filter(event_data=event_data, cutoff_frequency=20)
filter = KalmanFilter(event_data=event_data,c=0.01)
filter.Kalman_run()
# leaky_integrator(event_data,c=0.1)