import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from sklearn.metrics import jaccard_score
import os
import re

vid_name = 'test'
frame_num = 36


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

# Get recall between the ground truth silhouette (defined manually) and the predicted one
def get_recall(mask, otsu):
    TP = np.logical_and(mask == 1, otsu == 1).sum()
    FN = np.logical_and(mask == 1, otsu == 0).sum()
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recall

# Extract the predicted silhouettes using the 3 methods and compare them to the ground truth is the jaccard index
def compare_silhouette(vid_name,frame_num):

    video_path = f"../Videos/dataset/{vid_name}.mov"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = min(total_frames // 2, 100) if total_frames < 200 else min(total_frames, 200)

    cap.release() 


    back_gray, back_gray_blurred = background_gray(video_path, num_frames, 5)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()

    diff1 = np.abs(gray_frame - back_gray)
    diff1[diff1 >= 210] = 0
    diff1 = cv2.GaussianBlur(diff1, (15, 15), 0)
    diff1 = gamma_correct(diff1)

    otsu1 = method_1(diff1)
    otsu2 = method_2(diff1)
    otsu3 = method_3(diff1)

    mask = cv2.imread(f'../Mask/{vid_name}_frame_{frame_num}.png', cv2.IMREAD_GRAYSCALE)

    mask_bin = (mask > 0).astype(np.uint8)
    otsu1_bin = (otsu1 > 0).astype(np.uint8)
    otsu2_bin = (otsu2 > 0).astype(np.uint8)
    otsu3_bin = (otsu3 > 0).astype(np.uint8)



    jaccard1 = jaccard_score(mask_bin.flatten(), otsu1_bin.flatten())
    dice1 = (2 * jaccard1) / (1 + jaccard1)
    recall1 = get_recall(mask_bin, otsu1_bin)

    jaccard2 = jaccard_score(mask_bin.flatten(), otsu2_bin.flatten())
    dice2 = (2 * jaccard2) / (1 + jaccard2)
    recall2 = get_recall(mask_bin, otsu2_bin)

    jaccard3 = jaccard_score(mask_bin.flatten(), otsu3_bin.flatten())
    dice3 = (2 * jaccard3) / (1 + jaccard3)
    recall3 = get_recall(mask_bin, otsu3_bin)


    return [jaccard1, dice1, jaccard2, dice2, jaccard3, dice3]


# Perform the comparison for all frames manually segmented
jaccard_list = []

mask_folder = "../Mask"
files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])

c = 0
for file in files:
    if file.endswith(".png"):
        name, _ = os.path.splitext(file)
        match = re.match(r"(.*)_frame_(\d+)", name)
        if match:
            vid_name = match.group(1)
            frame_num = int(match.group(2))

            jaccard = compare_silhouette(vid_name,frame_num)
            jaccard_list.append(jaccard)
            print(c, file, jaccard)
            c +=1
            

print(jaccard_list)

jaccard_array = np.array(jaccard_list)
means = np.mean(jaccard_array, axis=0)
labels = ['Jaccard M1', 'Dice M1', 'Jaccard M2', 'Dice M2', 'Jaccard M3', 'Dice M3']



# Display results
plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
bar = plt.bar(x, means, color='skyblue', label='Mean')

for i in range(len(labels)):
    jitter = 0.1 * (np.random.rand(len(jaccard_array)) - 0.5)  
    plt.scatter(np.full(len(jaccard_array), x[i]) + jitter, jaccard_array[:, i], color='black', alpha=0.6, s=20)

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Mean and Individual Values of Jaccard & Dice Scores")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

