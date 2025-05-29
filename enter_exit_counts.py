import cv2
import argparse

from tracker import IOUTracker
from fetch_frames import VIDEO_PATH, FRAMES_DIR
from glob import glob
from groundlight import Groundlight, Detector

def process_frames(detector_name, output_video):
    # get the detector
    gl = Groundlight()
    detector: Detector = gl.get_detector_by_name(
        name=detector_name,
    )
    # load frames and submit to detector to get rois
    frame_files = sorted(glob(f"{FRAMES_DIR}/*.jpg"))[::3] # every 3rd frame to reduce load
    frames_rois = []
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        # Resize frame to 640x480 for faster processing
        frame = cv2.resize(frame, (640, 480))
        # Submit image to detector
        iq = gl.ask_ml(detector=detector, image=frame)
        frames_rois.append(iq.rois)

    # initialize the tracker
    tracker = IOUTracker()
    # specify a virtual line in the video in normalized coordinates
    line_start = (0.54, 0.99)
    line_end = (0.35, 0.70)

    # write a video with the detections
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(f"{VIDEO_PATH}/{output_video}", fourcc, 10, (640, 480))
    for frame_file, frame_rois in zip(frame_files, frames_rois):
        detections = [(roi.geometry.left, roi.geometry.top, roi.geometry.right, roi.geometry.bottom) for roi in frame_rois]
        tracker.update(detections)
        tracker.count_crossing(line_start, line_end)
        
        frame = cv2.imread(frame_file)
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        for roi in frame_rois:
                left, top, right, bottom = roi.geometry.left, roi.geometry.top, roi.geometry.right, roi.geometry.bottom
                left = int(left * w)
                right = int(right * w)
                top = int(top * h)
                bottom = int(bottom * h)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{roi.score:.2f}", (left, top+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
        # draw virtual line
        cv2.line(frame, (int(line_start[0]*w), int(line_start[1]*h)), (int(line_end[0]*w), int(line_end[1]*h)), (0, 0, 255), 2)
        # draw counts
        cv2.putText(frame, f"Total: {len(frame_rois)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
        cv2.putText(frame, f"Entered: {tracker.enter_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
        cv2.putText(frame, f"Exited: {tracker.exit_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        video_out.write(frame)
    video_out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process frames with Groundlight detector and create output video")
    parser.add_argument(
        "--name", "-n",
        type=str, 
        default="employees_with_groundlight_tshirt",
        help="Name of the Groundlight detector to use (default: employees_with_groundlight_tshirt)",
    )
    parser.add_argument(
        "--output-video", "-o",
        type=str,
        default="output.mp4",
        help="Output video name (default: output.mp4)",
    )
    
    args = parser.parse_args()
    process_frames(args.name, args.output_video)