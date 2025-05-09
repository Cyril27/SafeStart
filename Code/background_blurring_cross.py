
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from sklearn.metrics import jaccard_score

# Define the video and the frame of interest
vid_name = 'sub_1D'
frame_num = 35
video_path = f"../Videos/Dataset/{vid_name}.mov"

# Get video properties
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = min(total_frames // 2, 100) if total_frames < 200 else min(total_frames, 200)

cap.release()  


# Compute the background out of grayscale frames (blurred or not)
def background_gray(video_path, num_frames, kernel_size):
    cap = cv2.VideoCapture(video_path)  # Reopen video
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


back_gray, back_gray_blurred = background_gray(video_path, num_frames, 5)


# Get the per pixer mean error map by considering each blurred/ not blurred frame blurred and the blurred/ not blurred background
def per_pixel_mean_error_map(video_path, num_frames, background, im_type, blur_bool):
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
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

        elif im_type == 'hsv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        elif im_type == 'rgb':
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if blur_bool:
            diff = np.abs(blurred_frame.astype(np.float32) - background.astype(np.float32))
        else: 
            diff = np.abs(frame.astype(np.float32) - background.astype(np.float32))
        error_accumulator += diff
        count += 1

    cap.release()
    return error_accumulator / count if count > 0 else None



error_gray_im = per_pixel_mean_error_map(video_path, num_frames, back_gray, 'gray', False)
error_gray_blurred_im = per_pixel_mean_error_map(video_path, num_frames, back_gray_blurred, 'gray', False)
error_gray_im_blurred = per_pixel_mean_error_map(video_path, num_frames, back_gray, 'gray', True)
error_gray_blurred_im_blurred = per_pixel_mean_error_map(video_path, num_frames, back_gray_blurred, 'gray', True)


#Get the mean error over the per pixel error map
def get_error_magnitude(error_map):
    if len(error_map.shape) == 2: 
        return error_map, np.mean(np.abs(error_map))
    elif len(error_map.shape) == 3:  
        return np.sum(np.abs(error_map), axis=2), np.mean(np.abs(error_map))
        
    else:
        raise ValueError("Unexpected shape in error_map")


map1, err1 = get_error_magnitude(error_gray_im)
map2, err2 = get_error_magnitude(error_gray_blurred_im)
map3, err3 = get_error_magnitude(error_gray_im_blurred)
map4, err4 = get_error_magnitude(error_gray_blurred_im_blurred)

# Plot error maps 
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(map1, cmap='gray')
plt.title('Image/Background')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(map2, cmap='gray')
plt.title('Image/Background blurred')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(map3, cmap='gray')
plt.title('Image blurred/Background')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(map4, cmap='gray')
plt.title('Image blurred/Background blurred')
plt.axis('off')
plt.suptitle("Background investigation", fontsize=14)
plt.tight_layout()
#plt.show()


# Select grayscale frame of interest blurred/not blurred
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_frame_smoothed = cv2.GaussianBlur(gray_frame, (15, 15), 0)

cap.release()

# Background subtraction
diff1 = np.abs(gray_frame - back_gray)
diff2 = np.abs(gray_frame - back_gray_blurred)
diff3 = np.abs(gray_frame_smoothed - back_gray)
diff4 = np.abs(gray_frame_smoothed - back_gray_blurred)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(diff1)
plt.title('Image - Background')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(diff2)
plt.title('Image- Background blurred')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(diff3)
plt.title('Image blurred - Background')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(diff4)
plt.title('Image blurred - Background blurred')
plt.axis('off')
plt.suptitle("Background subtraction", fontsize=14)
plt.tight_layout()

# Thresholding of artifacts
diff1[diff1 >= 210] = 0
diff2[diff2 >= 210] = 0
diff3[diff3 >= 210] = 0
diff4[diff4 >= 210] = 0

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(diff1)
plt.title('Image - Background')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(diff2)
plt.title('Image- Background blurred')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(diff3)
plt.title('Image blurred - Background')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(diff4)
plt.title('Image blurred - Background blurred')
plt.axis('off')
plt.suptitle("Artifacts thresholding", fontsize=14)
plt.tight_layout()

# Gaussian blurring
diff1 = cv2.GaussianBlur(diff1, (15, 15), 0)
diff2 = cv2.GaussianBlur(diff2, (15, 15), 0)
diff3 = cv2.GaussianBlur(diff3, (15, 15), 0)
diff4 = cv2.GaussianBlur(diff4, (15, 15), 0)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(diff1)
plt.title('Image - Background')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(diff2)
plt.title('Image- Background blurred')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(diff3)
plt.title('Image blurred - Background')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(diff4)
plt.title('Image blurred - Background blurred')
plt.axis('off')
plt.suptitle("Gaussian blurring", fontsize=14)
plt.tight_layout()


# Perform gamma correction to increase dark regions contrast
def gamma_correct(frame):
    gamma = 0.3
    diff_frame = frame / 255.0  
    diff_frame = np.power(diff_frame, gamma)

    a = 1
    diff_frame = a * diff_frame  
    diff_frame = np.clip(diff_frame, 0, 1) 
    diff_frame = (diff_frame * 255).astype(np.uint8) 
    return diff_frame

diff1 = gamma_correct(diff1)
diff2 = gamma_correct(diff2)
diff3 = gamma_correct(diff3)
diff4 = gamma_correct(diff4)



plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(diff1)
plt.title('Image - Background')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(diff2)
plt.title('Image- Background blurred')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(diff3)
plt.title('Image blurred - Background')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(diff4)
plt.title('Image blurred - Background blurred')
plt.axis('off')
plt.suptitle("Gamma correction", fontsize=14)
plt.tight_layout()
plt.show()

# disk_kernel = morphology.disk(5)

# dilation = morphology.dilation(diff1, footprint=disk_kernel)
# dilation = morphology.dilation(dilation, footprint=disk_kernel)
# dilation1 = morphology.dilation(dilation, footprint=disk_kernel)

# dilation = morphology.dilation(diff2, footprint=disk_kernel)
# dilation = morphology.dilation(dilation, footprint=disk_kernel)
# dilation2 = morphology.dilation(dilation, footprint=disk_kernel)

# dilation = morphology.dilation(diff3, footprint=disk_kernel)
# dilation = morphology.dilation(dilation, footprint=disk_kernel)
# dilation3 = morphology.dilation(dilation, footprint=disk_kernel)

# dilation = morphology.dilation(diff4, footprint=disk_kernel)
# dilation = morphology.dilation(dilation, footprint=disk_kernel)
# dilation4 = morphology.dilation(dilation, footprint=disk_kernel)

# plt.figure(figsize=(12, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(dilation1)
# plt.title('im + back')
# plt.axis('off')
# plt.subplot(2, 2, 2)
# plt.imshow(dilation2)
# plt.title('im + back blurred')
# plt.axis('off')
# plt.subplot(2, 2, 3)
# plt.imshow(dilation3)
# plt.title('im blurred + back ')
# plt.axis('off')
# plt.subplot(2, 2, 4)
# plt.imshow(dilation4)
# plt.title('im blurred + back blurred')
# plt.axis('off')
# plt.tight_layout()
# #plt.show()


# val, otsu1 = cv2.threshold(dilation1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# val, otsu2 = cv2.threshold(dilation2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# val, otsu3 = cv2.threshold(dilation3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# val, otsu4 = cv2.threshold(dilation4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)




# plt.figure(figsize=(12, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(otsu1)
# plt.title('im + back')
# plt.axis('off')
# plt.subplot(2, 2, 2)
# plt.imshow(otsu2)
# plt.title('im + back blurred')
# plt.axis('off')
# plt.subplot(2, 2, 3)
# plt.imshow(otsu3)
# plt.title('im blurred + back ')
# plt.axis('off')
# plt.subplot(2, 2, 4)
# plt.imshow(otsu4)
# plt.title('im blurred + back blurred')
# plt.axis('off')
# plt.tight_layout()
# #plt.show()

# filled_mask = cv2.imread(f'../Mask/{vid_name}_frame_{frame_num}.png', cv2.IMREAD_GRAYSCALE)
# _, filled_mask = cv2.threshold(filled_mask, 127, 255, cv2.THRESH_BINARY)

# filled_mask_bin = filled_mask // 255  # Convert to 0 and 1

# # List of Otsu thresholded masks
# otsu_masks = [otsu1, otsu2, otsu3, otsu4]
# titles = ['im + back', 'im + back blurred', 'im blurred + back', 'im blurred + back blurred']

# plt.figure(figsize=(12, 10))

# for i, (otsu, title) in enumerate(zip(otsu_masks, titles)):
#     # Ensure binary and same size
#     _, otsu_bin = cv2.threshold(otsu, 127, 255, cv2.THRESH_BINARY)
#     otsu_bin = cv2.resize(otsu_bin, (filled_mask.shape[1], filled_mask.shape[0]))
#     otsu_bin = otsu_bin // 255

#     # Compute Jaccard Score (flattened arrays)
#     jaccard = jaccard_score(filled_mask_bin.flatten(), otsu_bin.flatten())

#     # Create overlay image: green for GT, red for prediction, yellow = overlap
#     overlay = np.zeros((filled_mask.shape[0], filled_mask.shape[1], 3), dtype=np.uint8)
#     overlay[np.where(filled_mask_bin == 1)] = [0, 255, 0]     # Ground truth = green
#     overlay[np.where(otsu_bin == 1)] = [255, 0, 0]            # Prediction = red
#     overlay[np.where((filled_mask_bin + otsu_bin) == 2)] = [255, 255, 0]  # Overlap = yellow

#     # Plot
#     plt.subplot(2, 2, i+1)
#     plt.imshow(overlay)
#     plt.title(f'{title}\nJaccard: {jaccard:.3f}')
#     plt.axis('off')
# plt.tight_layout()
# plt.show()





