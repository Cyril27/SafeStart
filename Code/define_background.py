import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the video and the frame of interest
vid_name = 'test'
frame_num = 36

video_path = f"../Videos/{vid_name}.mov"

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


back_rgb, back_rgb_blurred = background_rgb(video_path, num_frames, 3)
back_hsv, back_hsv_blurred = background_hsv(video_path, num_frames, 3)
back_gray, back_gray_blurred = background_gray(video_path, num_frames, 3)


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


error_gray = per_pixel_mean_error_map(video_path, num_frames, back_gray, 'gray')
error_hsv = per_pixel_mean_error_map(video_path, num_frames, back_hsv, 'hsv')
error_rgb = per_pixel_mean_error_map(video_path, num_frames, back_rgb, 'rgb')
error_gray_blurred = per_pixel_mean_error_map(video_path, num_frames, back_gray_blurred, 'gray')
error_hsv_blurred = per_pixel_mean_error_map(video_path, num_frames, back_hsv_blurred, 'hsv')
error_rgb_blurred = per_pixel_mean_error_map(video_path, num_frames, back_rgb_blurred, 'rgb')

#Get the mean error over the per pixel error map
def get_error_magnitude(error_map):
    if len(error_map.shape) == 2: 
        return error_map, np.mean(np.abs(error_map))
    elif len(error_map.shape) == 3:  
        return np.sum(np.abs(error_map), axis=2), np.mean(np.abs(error_map))
        
    else:
        raise ValueError("Unexpected shape in error_map")

# Prepare grayscale error maps
vis_gray, error_gray = get_error_magnitude(error_gray)
vis_hsv, error_hsv = get_error_magnitude(error_hsv)
vis_rgb, error_rgb = get_error_magnitude(error_rgb)
vis_gray_blurred, error_gray_blurred = get_error_magnitude(error_gray_blurred)
vis_hsv_blurred, error_hsv_blurred = get_error_magnitude(error_hsv_blurred)
vis_rgb_blurred, error_rgb_blurred = get_error_magnitude(error_rgb_blurred)


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

# Get the pSNR of the background and the reference frame (first frame of the video)
def compute_psnr(reference, test_frame):
    reference = np.float32(reference)
    test_frame = np.float32(test_frame)

    mse = np.mean((reference - test_frame) ** 2)
    if mse == 0:
        return 100  

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


psnr_gray = compute_psnr(ref_gray_frame, back_gray)
psnr_hsv = compute_psnr(ref_hsv_frame, back_hsv)
psnr_rgb = compute_psnr(ref_rgb_frame, back_rgb)

psnr_gray_blurred = compute_psnr(ref_gray_frame_smoothed, back_gray_blurred)
psnr_hsv_blurred = compute_psnr(ref_hsv_frame_smoothed, back_hsv_blurred)
psnr_rgb_blurred = compute_psnr(ref_rgb_frame_smoothed, back_rgb_blurred)




# Display results
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(vis_gray, cmap='gray')
plt.title(f"Error: {error_gray:.2f}, pSNR: {psnr_gray:.2f}")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(vis_hsv, cmap='gray')
plt.title(f"Error: {error_hsv:.2f}, pSNR: {psnr_hsv:.2f}")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(vis_rgb, cmap='gray')
plt.title(f"Error: {error_rgb:.2f}, pSNR: {psnr_rgb:.2f}")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(vis_gray_blurred, cmap='gray')
plt.title(f"Error: {error_gray_blurred:.2f}, pSNR: {psnr_gray_blurred:.2f}")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(vis_hsv_blurred, cmap='gray')
plt.title(f"Error: {error_hsv_blurred:.2f}, pSNR: {psnr_hsv_blurred:.2f}")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(vis_rgb_blurred, cmap='gray')
plt.title(f"Error: {error_rgb_blurred:.2f}, pSNR: {psnr_rgb_blurred:.2f}")
plt.axis('off')

plt.tight_layout()
plt.show()


