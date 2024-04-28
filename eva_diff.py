#the begin of the two images folder must have the same begin, or there won't be aligned!
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)
def psnr(mse, max_value=1.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_value / np.sqrt(mse))

def calculate_metrics(image_dir1, image_dir2):
    mse_values = []
    psnr_values = []
    file_list1 = sorted(os.listdir(image_dir1))
    file_list2 = sorted(os.listdir(image_dir2))
    
    for i in range(1,min(len(file_list1), len(file_list2))):
        prev_image_path1 = os.path.join(image_dir1, file_list1[i-1])
        curr_image_path1 = os.path.join(image_dir1, file_list1[i])
        
        prev_image_path2 = os.path.join(image_dir2, file_list2[i-1])
        curr_image_path2 = os.path.join(image_dir2, file_list2[i])

        prev_image1 = cv2.imread(prev_image_path1, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        curr_image1 = cv2.imread(curr_image_path1, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        prev_image2 = cv2.imread(prev_image_path2, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        curr_image2 = cv2.imread(curr_image_path2, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Check if images were loaded successfully
        if prev_image1 is None or curr_image1 is None or prev_image2 is None or curr_image2 is None:
            print(f"Failed to load images: {prev_image_path1}, {curr_image_path1}, {prev_image_path2}, {curr_image_path2}")
            continue

        # Calculate MSE and PSNR between difference images
        diff_image1 = curr_image1 - prev_image1
        diff_image2 = curr_image2 - prev_image2
        mse_value = mse(diff_image1, diff_image2)
        psnr_value = psnr(mse_value)


        mse_values.append(mse_value)
        psnr_values.append(psnr_value)

    return mse_values, psnr_values

def calculate_metrics2(image_dir1, image_dir2):
    mse_values = []
    psnr_values = []
    
    file_list1 = sorted(os.listdir(image_dir1))
    file_list2 = sorted(os.listdir(image_dir2))
    
    for i in range(2, min(len(file_list1), len(file_list2))):
        prev_image_path1 = os.path.join(image_dir1, file_list1[i-1])
        curr_image_path1 = os.path.join(image_dir1, file_list1[i])
        
        prev_image_path2 = os.path.join(image_dir2, file_list2[i-1])
        curr_image_path2 = os.path.join(image_dir2, file_list2[i])
        # prev_image_path2 = os.path.join(image_dir2, file_list2[i-2])
        # curr_image_path2 = os.path.join(image_dir2, file_list2[i-1])

        prev_image1 = cv2.imread(prev_image_path1, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        curr_image1 = cv2.imread(curr_image_path1, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        prev_image2 = plt.imread(prev_image_path2).astype(np.float32)
        curr_image2 = plt.imread(curr_image_path2).astype(np.float32)
        prev_image2 = np.mean(prev_image2[:, :, :3], axis=2)
        curr_image2 = np.mean(curr_image2[:, :, :3], axis=2)

        # Check if images were loaded successfully
        if prev_image1 is None or curr_image1 is None or prev_image2 is None or curr_image2 is None:
            print(f"Failed to load images: {prev_image_path1}, {curr_image_path1}, {prev_image_path2}, {curr_image_path2}")
            continue


        # Calculate MSE and PSNR between difference images
        diff_image1 = curr_image1 - prev_image1
        diff_image2 = curr_image2 - prev_image2
        diff_image1_normalized = cv2.normalize(diff_image1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        diff_image2_normalized = cv2.normalize(diff_image2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 保存图像
        cv2.imwrite('diff_image1.png', diff_image1_normalized)
        cv2.imwrite('diff_image2.png', diff_image2_normalized)
        mse_value = mse(diff_image1, diff_image2)
        psnr_value = psnr(mse_value)


        mse_values.append(mse_value)
        psnr_values.append(psnr_value)

    return mse_values, psnr_values
# Directories for images
# true_image_dir = "data/mic_colmap_easy/images"
# e2_image_dir = "data/mic_colmap_easy/images_e2"
# filter_image_dir = "data/mic_colmap_easy/output_images"
# # filter_image_dir = "data/mic_colmap_easy/output_images_pure"

# true_image_dir = "data/boxes_6dof/images"
true_image_dir = "data/boxes_6dof/output_images_pure"
# e2_image_dir = "data/boxes_6dof/images_e2"
# filter_image_dir = "data/boxes_6dof/output_images_com"
filter_image_dir = "data/boxes_6dof/output_images"

# Calculate metrics for e2 images
# e2_mse_values, e2_psnr_values = calculate_metrics(true_image_dir, e2_image_dir)
# avg_e2_mse = np.mean(e2_mse_values)
# avg_e2_psnr = np.mean(e2_psnr_values)

# print("Metrics for e2 images:")
# print("Average MSE:", avg_e2_mse)
# print("Average PSNR:", avg_e2_psnr)
# Calculate metrics for filter images
filter_mse_values, filter_psnr_values = calculate_metrics2(true_image_dir, filter_image_dir)

# Calculate average MSE and PSNR

avg_filter_mse = np.mean(filter_mse_values)
avg_filter_psnr = np.mean(filter_psnr_values)


print("/nMetrics for filter images:")
print("Average MSE:", avg_filter_mse)
print("Average PSNR:", avg_filter_psnr)