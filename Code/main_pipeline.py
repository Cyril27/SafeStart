import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import convolve
import os
import pandas as pd




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

def gamma_correct(frame):
    gamma = 0.3
    diff_frame = frame / 255.0  
    diff_frame = np.power(diff_frame, gamma)

    a = 1
    diff_frame = a * diff_frame  
    diff_frame = np.clip(diff_frame, 0, 1) 
    diff_frame = (diff_frame * 255).astype(np.uint8) 
    return diff_frame


# Method 1 to define the silhouette  
def method_1(im):

    disk_kernel = morphology.disk(5)

    dilation = morphology.dilation(im, footprint=disk_kernel)
    dilation = morphology.dilation(dilation, footprint=disk_kernel)
    dilation = morphology.dilation(dilation, footprint=disk_kernel)

    kernel = morphology.disk(3)
    dilation = morphology.erosion(dilation, footprint=kernel)

    val, otsu = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu

# Method 2 to define the silhouette  
def method_2(im):

    vertical_kernel = np.ones((9, 3), np.uint8)
    v_dilation = morphology.closing(im, footprint=vertical_kernel)
    v_dilation2 = morphology.closing(v_dilation, footprint=vertical_kernel)
    v_dilation3 = morphology.closing(v_dilation2, footprint=vertical_kernel)

    horizontal_kernel = np.ones((3, 9), np.uint8)
    h_dilation = morphology.closing(im, footprint=horizontal_kernel)
    h_dilation2 = morphology.closing(h_dilation, footprint=horizontal_kernel)
    h_dilation3 = morphology.closing(h_dilation2, footprint=horizontal_kernel)

    val, v_otsu_thresh_dil3 = cv2.threshold(v_dilation3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    val, h_otsu_thresh_dil3 = cv2.threshold(h_dilation3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((9, 9), np.uint8)
    comb = np.logical_or(v_otsu_thresh_dil3, h_otsu_thresh_dil3).astype(np.uint8)
    comb = morphology.dilation(comb, footprint=kernel)
    comb = morphology.dilation(comb, footprint=kernel)
    comb = morphology.dilation(comb, footprint=kernel)
    comb = comb * 255
    return comb

# Method 3 to define the silhouette  
def method_3(im):
    vertical_kernel = np.ones((25, 3), np.uint8)
    v_dilation = morphology.closing(im, footprint=vertical_kernel)
    v_dilation2 = morphology.closing(v_dilation, footprint=vertical_kernel)
    v_dilation3 = morphology.closing(v_dilation2, footprint=vertical_kernel)

    horizontal_kernel = np.ones((3, 25), np.uint8)
    h_dilation = morphology.closing(im, footprint=horizontal_kernel)
    h_dilation2 = morphology.closing(h_dilation, footprint=horizontal_kernel)
    h_dilation3 = morphology.closing(h_dilation2, footprint=horizontal_kernel)

    combined_dilation = np.maximum(v_dilation3, h_dilation3)
    val, combined_thresh = cv2.threshold(combined_dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((21, 21), np.uint8)
    comb = morphology.dilation(combined_thresh, footprint=kernel)
    comb = morphology.dilation(comb, footprint=kernel)
    comb = morphology.dilation(comb, footprint=kernel)
    kernel = morphology.disk(15)
    comb = morphology.erosion(comb, footprint=kernel)
    comb = morphology.erosion(comb, footprint=kernel)
    comb = comb * 255
    return comb


# Extract metrics from the computed skeleton 
def analyze_skeleton_structure(skeleton):
    if skeleton.sum() == 0:
        return {
            'total_length': 0,
            'endpoints': 0,
            'branchpoints': 0,
            'approx_branches': 0,
            'avg_branch_length': 0.0
        }

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    
    # Add strong center weight to distinguish self from neighbors
    neighbor_sum = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    mask = skeleton.astype(bool)

    # Count actual neighbors (exclude center by subtracting 10)
    num_neighbors = (neighbor_sum - 10)[mask]

    endpoints = np.sum(num_neighbors == 1)
    branchpoints = np.sum(num_neighbors >= 3)
    total_length = np.sum(mask)
    approx_num_branches = max(endpoints // 2, 1)
    avg_branch_length = total_length / approx_num_branches

    return {
        'total_length': int(total_length),
        'endpoints': int(endpoints),
        'branchpoints': int(branchpoints),
        'approx_branches': int(approx_num_branches),
        'avg_branch_length': float(avg_branch_length)
    }


# GEt the skeleton for a given method, apply pruning and extract metrics
def skeleton_per_method(diff, method):

    otsu_mask = method(diff)
    binary_mask = (otsu_mask > 0).astype(np.uint8)

    skeleton = skeletonize(binary_mask)
    skeleton = remove_small_objects(skeleton, min_size=300, connectivity=2)

    metrics = analyze_skeleton_structure(skeleton)
    skeleton = skeleton.astype(np.uint8) * 255

    return skeleton, metrics



# Compute the skeleton and the metrics for several frames of a given video
def get_skeleton(video_path):
    cap = cv2.VideoCapture(video_path)
    total_res = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(total_frames // 2, 100) if total_frames < 200 else min(total_frames, 200)

    background, _ = background_gray(video_path, num_frames, 5)

    for frame_num in range(50, 80, 5):  
        print(f"Processing frame: {frame_num}")
        res = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray - background)
        diff[diff >= 210] = 0  
        diff = cv2.GaussianBlur(diff, (15, 15), 0)
        diff = gamma_correct(diff)

        ske1, metrics1 = skeleton_per_method(diff, method_1)
        ske2, metrics2 = skeleton_per_method(diff, method_2)
        ske3, metrics3 = skeleton_per_method(diff, method_3)

        res.extend(metrics1.values())
        res.extend(metrics2.values())
        res.extend(metrics3.values())
        
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(ske1)
        # plt.title("Method 1")
        # plt.subplot(1, 3, 2)
        # plt.imshow(ske2)
        # plt.title("Method 2")
        # plt.subplot(1, 3, 3)
        # plt.imshow(ske3)
        # plt.title("Method 3")
        # plt.show()
        total_res.append(res)

    cap.release()
    return total_res



# Loop over the videos of the dataset
def process_videos_in_folder(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mov', '.avi'))]
    results = []
    count = 0

    for video_file in video_files:
        print(video_file)
        video_path = os.path.join(folder_path, video_file)

        video_res = get_skeleton(video_path)
        for row in video_res:
            results.append(row)


        count +=1
        if count >10:
            break
        

    return results

folder_path = "../Videos/dataset"
results = process_videos_in_folder(folder_path)


#Save results in a df 
def save_results_to_csv(results, output_path):
    # Convert to DataFrame
    columns = [
        "total_length_1", "endpoints_1", "branchpoints_1", "approx_branches_1", "avg_branch_length_1",
        "total_length_2", "endpoints_2", "branchpoints_2", "approx_branches_2", "avg_branch_length_2", 
        "total_length_3", "endpoints_3", "branchpoints_3", "approx_branches_3", "avg_branch_length_3" 
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    return df 

#df = save_results_to_csv(results, '../skeleton_metrics.csv')