import os
import cv2

VIDEO_PATH = "./video/"
FRAMES_DIR = f"{VIDEO_PATH}/frames"

# create directory to save frames
os.makedirs(FRAMES_DIR, exist_ok=True)
# open mp4 file and read frames
cap = cv2.VideoCapture(f"{VIDEO_PATH}/counting_video.mp4")

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{FRAMES_DIR}/{frame_num:04d}.jpg", frame)
    frame_num += 1