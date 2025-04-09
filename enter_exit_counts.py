import cv2
from tracker import IOUTracker
from fetch_frames import VIDEO_PATH, FRAMES_DIR
from glob import glob
from groundlight import Groundlight, Detector

# get the detector
gl = Groundlight()
detector: Detector = gl.get_or_create_detector(
    name="employees_with_groundlight_tshirt",
    query="Label each employee with Groundlight T-shirt in the image",
)
# load frames and submit to detector to get rois
frame_files = sorted(glob(f"{FRAMES_DIR}/*.jpg"))[::3] # every 3rd frame to reduce load
frames_rois = []
for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    # Resize frame to 640x480 for faster processing
    frame = cv2.resize(frame, (640, 480))
    # Submit image to detector
    iq = gl.submit_image_query(detector=detector, image=frame)
    frames_rois.append(iq.rois)

# initialize the tracker
tracker = IOUTracker()
# specify a virtual line in the video in normalized coordinates
line_start = (0.54, 0.99)
line_end = (0.35, 0.70)

# write a video with the detections
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(f"{VIDEO_PATH}/output.mp4", fourcc, 10, (640, 480))
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
		cv2.putText(frame, f"Entered: {tracker.enter_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
		cv2.putText(frame, f"Exited: {tracker.exit_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		video_out.write(frame)
video_out.release()
