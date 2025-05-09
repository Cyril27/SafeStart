import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# Compute the background out of rgb frames (blurred or not)
def background_rgb(video_path, num_frames, kernel_size):
    cap = cv2.VideoCapture(video_path) 
    frames = []
    frames_blurred = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {i}")
            break
        
        blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        frames.append(frame)
        frames_blurred.append(blurred_frame)

    cap.release()

    if len(frames) == 0:
        print("Error: No frames captured for background modeling (HSV).")
        return None

    background_model = np.median(np.array(frames, dtype=np.float32), axis=0).astype(np.uint8)
    background_model_blurred = np.median(np.array(frames_blurred, dtype=np.float32), axis=0).astype(np.uint8)

    return background_model, background_model_blurred

# Compute the background out of hsv frames (blurred or not)
def background_hsv(video_path, num_frames, kernel_size):
    cap = cv2.VideoCapture(video_path) 
    frames = []
    frames_blurred = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {i}")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred_frame = cv2.GaussianBlur(hsv_frame, (kernel_size, kernel_size), 0)
        frames.append(hsv_frame)
        frames_blurred.append(blurred_frame)

    cap.release()

    if len(frames) == 0:
        print("Error: No frames captured for background modeling (HSV).")
        return None

    background_model = np.median(np.array(frames, dtype=np.float32), axis=0).astype(np.uint8)
    background_model_blurred = np.median(np.array(frames_blurred, dtype=np.float32), axis=0).astype(np.uint8)

    return background_model, background_model_blurred

# Compute the background out of grayscale frames (blurred or not)
def background_gray(video_path, num_frames, kernel_size):
    cap = cv2.VideoCapture(video_path) 
    frames = []
    frames_blurred = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {i}")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_smoothed = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
        
        frames.append(gray_frame)
        frames_blurred.append(gray_frame_smoothed)

    cap.release()

    if len(frames) == 0:
        print("Error: No frames captured for background modeling (Gray).")
        return None

    background_model = np.median(np.array(frames), axis=0).astype(np.uint8)
    background_model_blurred = np.median(np.array(frames_blurred), axis=0).astype(np.uint8)

    return background_model, background_model_blurred

# Get the per pixer mean error map by considering each blurred frame with the blurred background
# or not blurred frame with the not blurred background
def per_pixel_mean_error_map(video_path, num_frames, background, im_type):
    cap = cv2.VideoCapture(video_path)
    error_accumulator = np.zeros_like(background, dtype=np.float32)
    count = 0

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {i}")
            break

        if im_type == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        elif im_type == 'hsv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        elif im_type == 'rgb':
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        diff = np.abs(blurred_frame.astype(np.float32) - background.astype(np.float32))
        error_accumulator += diff
        count += 1

    cap.release()
    return error_accumulator / count if count > 0 else None

#Get the mean error over the per pixel error map
def get_error_magnitude(error_map):
    """Returns grayscale version of error (for multi-channel, reduce to magnitude)."""
    if len(error_map.shape) == 2: 
        return error_map, np.mean(np.abs(error_map))
    elif len(error_map.shape) == 3:  
        return np.sum(np.abs(error_map), axis=2), np.mean(np.abs(error_map))
        
    else:
        raise ValueError("Unexpected shape in error_map")

# Get the pSNR of the background and the reference frame (first frame of the video)
def compute_psnr(reference, test_frame):
    # Convert both to float32 for precision in calculations
    reference = np.float32(reference)
    test_frame = np.float32(test_frame)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((reference - test_frame) ** 2)
    if mse == 0:
        return 100  # Infinite pSNR (perfect match)

    # Maximum pixel value
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_errors_and_psnr(video_path, num_frames):
    # Compute background models
    back_rgb, back_rgb_blurred = background_rgb(video_path, num_frames, 3)
    back_hsv, back_hsv_blurred = background_hsv(video_path, num_frames, 3)
    back_gray, back_gray_blurred = background_gray(video_path, num_frames, 3)

    # Compute errors
    error_gray = per_pixel_mean_error_map(video_path, num_frames, back_gray, 'gray')
    error_hsv = per_pixel_mean_error_map(video_path, num_frames, back_hsv, 'hsv')
    error_rgb = per_pixel_mean_error_map(video_path, num_frames, back_rgb, 'rgb')
    error_gray_blurred = per_pixel_mean_error_map(video_path, num_frames, back_gray_blurred, 'gray')
    error_hsv_blurred = per_pixel_mean_error_map(video_path, num_frames, back_hsv_blurred, 'hsv')
    error_rgb_blurred = per_pixel_mean_error_map(video_path, num_frames, back_rgb_blurred, 'rgb')

    # Prepare grayscale error maps
    vis_gray, error_gray = get_error_magnitude(error_gray)
    vis_hsv, error_hsv = get_error_magnitude(error_hsv)
    vis_rgb, error_rgb = get_error_magnitude(error_rgb)
    vis_gray_blurred, error_gray_blurred = get_error_magnitude(error_gray_blurred)
    vis_hsv_blurred, error_hsv_blurred = get_error_magnitude(error_hsv_blurred)
    vis_rgb_blurred, error_rgb_blurred = get_error_magnitude(error_rgb_blurred)

    # Compute pSNR
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, ref_frame = cap.read()

    ref_gray_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_gray_frame_smoothed = cv2.GaussianBlur(ref_gray_frame, (15, 15), 0)
    ref_hsv_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2HSV)
    ref_hsv_frame_smoothed = cv2.GaussianBlur(ref_hsv_frame, (15, 15), 0)
    ref_rgb_frame = ref_frame
    ref_rgb_frame_smoothed = cv2.GaussianBlur(ref_rgb_frame, (15, 15), 0)

    cap.release()

    psnr_gray = compute_psnr(ref_gray_frame, back_gray)
    psnr_hsv = compute_psnr(ref_hsv_frame, back_hsv)
    psnr_rgb = compute_psnr(ref_rgb_frame, back_rgb)
    psnr_gray_blurred = compute_psnr(ref_gray_frame_smoothed, back_gray_blurred)
    psnr_hsv_blurred = compute_psnr(ref_hsv_frame_smoothed, back_hsv_blurred)
    psnr_rgb_blurred = compute_psnr(ref_rgb_frame_smoothed, back_rgb_blurred)

    return [
        error_gray, error_hsv, error_rgb, error_gray_blurred, error_hsv_blurred, error_rgb_blurred,
        psnr_gray, psnr_hsv, psnr_rgb, psnr_gray_blurred, psnr_hsv_blurred, psnr_rgb_blurred
    ]

#Get the avergae grayscale error and pSNR for all videos of the dataset
def process_videos_in_folder(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mov', '.avi'))]
    results = []

    for video_file in video_files:
        print(video_file)
        video_path = os.path.join(folder_path, video_file)

        num_frames = 100  
        errors_and_psnrs = compute_errors_and_psnr(video_path, num_frames)
        results.append([video_file] + errors_and_psnrs)

    return results

def save_results_to_csv(results, output_path):
    # Convert to DataFrame
    columns = [
        "Video", "Error_RGB", "Error_HSV", "Error_Gray", "Error_RGB_Blurred", "Error_HSV_Blurred", "Error_Gray_Blurred",
        "pSNR_RGB", "pSNR_HSV", "pSNR_Gray", "pSNR_RGB_Blurred", "pSNR_HSV_Blurred", "pSNR_Gray_Blurred"
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    return df 


# Define folder containing videos and output CSV path
folder_path = "../Videos/dataset"
output_csv_path = "../errors_and_psnr.csv"

# Process all videos in the folder
#results = process_videos_in_folder(folder_path)

# Save results to CSV
df = pd.read_csv("../errors_and_psnr.csv")


#Display results
error_columns = ['Error_Gray', 'Error_HSV', 'Error_RGB', 'Error_Gray_Blurred', 'Error_HSV_Blurred', 'Error_RGB_Blurred']
psnr_columns = ['pSNR_Gray', 'pSNR_HSV', 'pSNR_RGB', 'pSNR_Gray_Blurred', 'pSNR_HSV_Blurred', 'pSNR_RGB_Blurred']

error_means = df[error_columns].mean()
psnr_means = df[psnr_columns].mean()

bar_width = 0.10 
index = np.arange(len(error_columns)) * 0.2  

raw_error_values = df[error_columns].values.T  
raw_psnr_values = df[psnr_columns].values.T  #


fig, ax1 = plt.subplots(figsize=(6,5))
bars_error = ax1.bar(index, error_means, bar_width, label='Error', color='royalblue', edgecolor='black')
ax1.set_xlabel('Metric', fontsize=14)
ax1.set_ylabel('Error Mean Value', fontsize=14, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(index)
ax1.set_xticklabels(['Gray', 'HSV', 'RGB', 'Gray Blurred', 'HSV Blurred', 'RGB Blurred'])

for i, err_values in enumerate(raw_error_values):
    ax1.scatter([index[i]] * len(err_values), err_values,  color='white', edgecolor='black', alpha=0.7, label="Individual Error" if i == 0 else "")


ax1.legend(loc='upper right')
plt.title('Error Mean Values for Different Metrics', fontsize=16)

fig, ax2 = plt.subplots(figsize=(6,5))
bars_psnr = ax2.bar(index, psnr_means, bar_width, label='pSNR', color='royalblue', edgecolor='black')
ax2.set_xlabel('Metric', fontsize=14)
ax2.set_ylabel('pSNR Mean Value', fontsize=14, color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_xticks(index)
ax2.set_xticklabels(['Gray', 'HSV', 'RGB', 'Gray Blurred', 'HSV Blurred', 'RGB Blurred'])

for i, psnr_values in enumerate(raw_psnr_values):
    ax2.scatter([index[i]] * len(psnr_values), psnr_values, color='white', edgecolor='black', alpha=0.7, label="Individual pSNR" if i == 0 else "")

ax2.legend(loc='upper right')
plt.title('pSNR Mean Values for Different Metrics', fontsize=16)
plt.tight_layout()
plt.show()