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
from tqdm import tqdm

class KalmanFilter:
    def __init__(self,event_data,c = 0.1) -> None:
        self.event_data = event_data
        self.c = c
        self.noise_period_bound = 500*1e-6
        self.refractory_bound = 1e3 * 1e-6
        self.cutoff_frequency = 20.0
    def initialState(self,event_data,c=0.1):
    #     # Initialize an array of the same size as diff_state to store the event intensity corresponding to the minimum timestamp
        events, height, width = event_data.event_list, event_data.height, event_data.width
    #     max_count =  1.0*height * width
        init_state = np.zeros((height, width), dtype=np.float32)
    #     # Traverse all events
    #     i = 0
    #     for event in events:
    #         # Get the position and event intensity of the event
    #         x, y, p = event.x, event.y, event.p
            
    #         # If the current position is not initialized or the event intensity of the current event is greater than the recorded event intensity, update the event intensity
    #         # care about y and x!!!
    #         if init_state[y, x] == 0:
    #             init_state[y, x] = p*c
        
    #         # Check if all positions have been initialized
    #         # The probability of initializing all positions is very low, so it's fine to assign 0 to the background
    #         # Therefore, only take the first max_count events to initialize
    #         i = i + 1
    #         if i > max_count:
    #             break
        return init_state
    def getFk(self,x,y,t):
        #do not sure xk+1,thus Fk small but noise big
        # # 确定九个点的坐标
        # x_range = np.clip(np.arange(x-1, x+2), 0, self.time_surface.shape[1]-1)
        # y_range = np.clip(np.arange(y-1, y+2), 0, self.time_surface.shape[0]-1)
        
        # # 提取九个点的时间值
        # neighbor_times = self.time_surface[y_range[:, None], x_range]
        
        # # 计算事件时间与九个点时间的差值
        # time_diffs = t - neighbor_times
        
        # if np.all(time_diffs < self.noise_period_bound):
        #     return 1
        # else:
        #     return -1
        return 0.01
    def getRk(self,x,y,t):
        ##design it by the newest paper:
        #t microsecond ,us 1e6
        # sigma_ref = 0.01
        # sigma_proc = 0.0005
        # sigma_iso = 0.03
        
        #RK should around 1, may be I should not divide c
        sigma_ref = 0.01
        sigma_proc = 0.0005
        sigma_iso = 0.03
        t_diff = (t-self.time_surface[y,x])
        Q_proc = sigma_proc*(t_diff)

        # 确定九个点的坐标
        x_range = np.clip(np.arange(x-1, x+2), 0, self.time_surface.shape[1]-1)
        y_range = np.clip(np.arange(y-1, y+2), 0, self.time_surface.shape[0]-1)
        # 提取九个点的时间值
        neighbor_times = self.time_surface[y_range[:, None], x_range]
        # 获取最大值所在位置的索引
        max_index = np.unravel_index(np.argmax(neighbor_times), neighbor_times.shape)
        # 获取最大值
        t_Np = (t-neighbor_times[max_index])
        Q_iso = sigma_iso*(t-t_Np)

        if t_diff>self.refractory_bound:
            Q_ref = 0
        else:
            Q_ref = sigma_ref
        Q = Q_iso+Q_proc+Q_ref
        #original paper, e()from +c to -c, thus I need to divide c^2 in the result
        # return Q/(self.c*self.c)
        return Q
    def Kalman_run(self):
        print('Filtering with Kalman Filter, please wait...')
        event_data = self.event_data
        c = self.c
        events, height, width = event_data.event_list, event_data.height, event_data.width
        frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
        frame_timestamps = frame_timestamps
        events_per_frame = 2e4
        with Timer('Filtering'):
            image_state = np.zeros((height, width), dtype=np.float32)
            # diff_state = np.zeros((height, width), dtype=np.float32)
            diff_state = self.initialState(event_data,c)
            # diff_state = np.ones((height, width), dtype=np.float32)
            #share self.time_surface to save time
            self.time_surface = np.zeros((height, width), dtype=np.float32)
            # covariance_state = np.full((height, width),c*c, dtype=np.float32)
            covariance_state = np.zeros((height, width), dtype=np.float32)
            image_list = []
            frame_idx = 0
            max_frame_idx = len(frames) - 1
            for i, e in tqdm(enumerate(events), total=len(events)):
                if frame_idx < max_frame_idx:
                    # print(e.t)

                    if e.t >= frame_timestamps[frame_idx + 1]:
                        frame_idx += 1
                        # Process image_state and save to folder
                        processed_image_state = 2 ** image_state - 1
                        # processed_image_state = image_state
                        image_list.append(processed_image_state)
                        save_image(processed_image_state, frame_idx+2)

                # Kalman filter update step
                # You need to implement the Kalman filter equations here

                #parameter calculate
                Fk = self.getFk(e.x,e.y,e.t)
                # #Sk is wkvk, Qk is wkwk, R is vkvk, normally Rk> QK
                # but your xk+1 is totally blind, thus Qk more than c^2, while Rk can be small(from oberve data)
                Sk=0
                Qk=4/9*c*c *10000
                Rk=self.getRk(e.x,e.y,e.t)
                #update
                P = covariance_state[e.y, e.x]
                K = (Fk*P*1/c + Sk)/(1/c*P*1/c + Rk)
                diff_state[e.y, e.x] = Fk*diff_state[e.y, e.x] + K*(e.p - 1/c * diff_state[e.y, e.x])
                covariance_state[e.y, e.x] = (Fk-K*1/c)*P*(Fk-K*1/c)+Qk+K*Rk*K-2*Sk*K           
                # Keep track of image state and time surface
                self.time_surface[e.y, e.x] = e.t

                #update log-intensity through 
                image_state[e.y, e.x] = image_state[e.y, e.x] + diff_state[e.y, e.x]

        return image_list
    def hybrid_run(self):
        print('Filtering with Kalman Filter, please wait...')
        event_data = self.event_data
        c = 1
        events, height, width = event_data.event_list, event_data.height, event_data.width
        frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
        frame_timestamps = frame_timestamps
        events_per_frame = 2e4
        with Timer('Filtering'):
            image_state = np.zeros((height, width), dtype=np.float32)
            # diff_state = np.zeros((height, width), dtype=np.float32)
            diff_state = self.initialState(event_data,c)
            # diff_state = np.ones((height, width), dtype=np.float32)
            #share self.time_surface to save time
            self.time_surface = np.zeros((height, width), dtype=np.float32)
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
                        image_list.append(processed_image_state)
                        save_image(processed_image_state, frame_idx+2)

                # Kalman filter update step
                # You need to implement the Kalman filter equations here

                #parameter calculate
                Fk = self.getFk(e.x,e.y,e.t)
                # #Sk is wkvk, Qk is wkwk, R is vkvk, normally Rk> QK
                Sk=0
                Qk=0.1
                Rk=1
                #update
                P = covariance_state[e.y, e.x]
                K = (Fk*P*1/c + Sk)/(1/c*P*1/c + Rk)
                diff_state[e.y, e.x] = Fk*diff_state[e.y, e.x] + K*(e.p - 1/c * diff_state[e.y, e.x])
                covariance_state[e.y, e.x] = (Fk-K*1/c)*P*(Fk-K*1/c)+Qk+K*Rk*K-2*Sk*K           
                # Keep track of image state and time surface
                self.time_surface[e.y, e.x] = e.t

                #update log-intensity through 
                #use complementary filter
                # image_state[e.y, e.x] = image_state[e.y, e.x] + diff_state[e.y, e.x]
                beta = math.exp(-self.cutoff_frequency * (e.t - self.time_surface[e.y, e.x]))
                # image_state[e.y, e.x] = beta * image_state[e.y, e.x] \
                #                         + (1 - beta) * log_frame[e.y, e.x] + 0.1 * e.p
                image_state[e.y, e.x] = beta * image_state[e.y, e.x] \
                             + self.c * diff_state[e.y,e.x]

        return image_list
def save_image(image, index, folder_path="D:/2024/3DGS/PureEventFilter/data/ship_colmap_easy/output_images"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{index:05d}.png")
    # Saving image
    # print("output")
    # Convert to uint8 before saving with OpenCV
    # image_uint8 = (image * 255).astype(np.uint8)-
    # # Saving image with OpenCV
    # cv2.imwrite(file_path, image_uint8)
    plt.imsave(file_path, image, cmap='gray')