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
    
    for filename in sorted(os.listdir(image_dir1)):
        if filename.endswith('.png'):
            # Load images
            true_image_path = os.path.join(image_dir1, filename)
            filter_image_path = os.path.join(image_dir2, filename)

            true_image = cv2.imread(true_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            filter_image = cv2.imread(filter_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            # Check if images were loaded successfully
            if true_image is None or filter_image is None:
                print(f"Failed to load images: {true_image_path}, {filter_image_path}")
                continue

            # Calculate MSE and PSNR
            mse_value = mse(true_image, filter_image)
            psnr_value = psnr(mse_value)

            mse_values.append(mse_value)
            psnr_values.append(psnr_value)

    return mse_values, psnr_values
def calculate_metrics2(image_dir1, image_dir2):
    mse_values = []
    psnr_values = []
    
    for filename in sorted(os.listdir(image_dir2)):
        if filename.endswith('.png'):
            # Load images from image_dir1
            true_image_path = os.path.join(image_dir1, filename)
            true_image = cv2.imread(true_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            
            # Load images from image_dir2 using Matplotlib
            filter_image_path = os.path.join(image_dir2, filename)
            filter_image = plt.imread(filter_image_path).astype(np.float32)
            filter_image = np.mean(filter_image[:, :, :3], axis=2)
            # Check if images were loaded successfully
            if true_image is None or filter_image is None:
                print(f"Failed to load images: {true_image_path}, {filter_image_path}")
                continue

            # Calculate MSE and PSNR
            mse_value = mse(true_image, filter_image)
            psnr_value = psnr(mse_value)

            mse_values.append(mse_value)
            psnr_values.append(psnr_value)

    return mse_values, psnr_values
# Directories for images
true_image_dir = "data\mic_colmap_easy\images"
e2_image_dir = "data\mic_colmap_easy\images_e2"
filter_image_dir = "data\mic_colmap_easy\output_images"

# Calculate metrics for e2 images
e2_mse_values, e2_psnr_values = calculate_metrics(true_image_dir, e2_image_dir)

# Calculate metrics for filter images
filter_mse_values, filter_psnr_values = calculate_metrics2(true_image_dir, filter_image_dir)

# Calculate average MSE and PSNR
avg_e2_mse = np.mean(e2_mse_values)
avg_e2_psnr = np.mean(e2_psnr_values)
avg_filter_mse = np.mean(filter_mse_values)
avg_filter_psnr = np.mean(filter_psnr_values)

print("Metrics for e2 images:")
print("Average MSE:", avg_e2_mse)
print("Average PSNR:", avg_e2_psnr)

print("\nMetrics for filter images:")
print("Average MSE:", avg_filter_mse)
print("Average PSNR:", avg_filter_psnr)