import cv2
import numpy as np

# Parameters
vid_name = 'sub_6D'
frame_num = 55

# Path to the video
video_path = f"../Videos/dataset/{vid_name}.mov"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()

if not ret:
    print("Could not read frame.")
    exit()

# Create a black mask with the same height/width as the frame
mask = np.zeros(frame.shape[:2], dtype=np.uint8)

drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial mouse position

# Mouse callback function
def draw_mask(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(mask, (ix, iy), (x, y), 255, 5) 
            cv2.line(frame, (ix, iy), (x, y), (0, 255, 0), 2)  
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(mask, (ix, iy), (x, y), 255, 5)
        cv2.line(frame, (ix, iy), (x, y), (0, 255, 0), 2)


cv2.namedWindow('Draw Mask')
cv2.setMouseCallback('Draw Mask', draw_mask)

while True:
    cv2.imshow('Draw Mask', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  #
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)


        # Save the filled mask as a PNG image
        cv2.imwrite(f'../Mask/{vid_name}_frame_{frame_num}.png', filled_mask)
        print("Filled mask saved as filled_mask.png")

    elif key == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
