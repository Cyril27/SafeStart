import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects


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





# Load video
vid_name    = 'sub_6D'
video_path  = f"../Videos/Dataset/{vid_name}.mov"
output_path = f"../Videos/Results/{vid_name}_overlay_method1.mp4"
cap = cv2.VideoCapture(video_path)

fps         = cap.get(cv2.CAP_PROP_FPS)
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc      = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1', etc.

writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames   = min(total_frames // 2, 100) if total_frames < 200 else min(total_frames, 200)
background, _ = background_gray(video_path, num_frames, 5)


# Get the binary mask and the skeleton for the each frame and save it
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = np.abs(gray - background)
    diff[diff >= 210] = 0
    diff = cv2.GaussianBlur(diff, (15, 15), 0)
    diff = gamma_correct(diff)

    otsu_mask = method_2(diff)

    binary_mask = (otsu_mask > 0).astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8) * 255

    skeleton = skeletonize(binary_mask)
    skeleton = remove_small_objects(skeleton, min_size=300, connectivity=2)
    skeleton = skeleton.astype(np.uint8) * 255


    mask_color = np.zeros_like(frame)
    mask_color[:, :, 2] = binary_mask 

    alpha = 0.4
    overlay = cv2.addWeighted(frame, 1.0, mask_color, alpha, 0)
    overlay[skeleton > 0] = [0, 255, 0]

    writer.write(overlay)
    cv2.imshow("Overlay", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()