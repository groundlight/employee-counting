import os
import cv2
import argparse

VIDEO_PATH = "./video/"
FRAMES_DIR = f"{VIDEO_PATH}/frames"

def fetch_frames(video_name):
    # create directory to save frames
    os.makedirs(FRAMES_DIR, exist_ok=True)
    # open mp4 file and read frames
    cap = cv2.VideoCapture(f"{VIDEO_PATH}/{video_name}")

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{FRAMES_DIR}/{frame_num:04d}.jpg", frame)
        frame_num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video", "-v", 
                       default="counting_video.mp4", 
                       help="Name of the video file to get frames from (default: our sample video, counting_video.mp4)")
    
    args = parser.parse_args()
    fetch_frames(args.video)
