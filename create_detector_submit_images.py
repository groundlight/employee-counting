from groundlight import Groundlight, Detector, ImageQuery
from fetch_frames import FRAMES_DIR
from glob import glob
import cv2

SAMPLE_INTERVAL = 10 # sample every 10th frame

gl = Groundlight()
detector: Detector = gl.get_or_create_detector(
    name="employees_with_groundlight_tshirt",
    query="Label each employee with Groundlight T-shirt in the image",
)

frame_files = sorted(glob(f"{FRAMES_DIR}/*.jpg"))

# submit every SAMPLE_INTERVAL frame to the detector in the first half of the video
for frame_file in frame_files[:len(frame_files)//2:SAMPLE_INTERVAL]:
	# load frame
	frame = cv2.imread(frame_file)
	# resize frame to 640x480 for faster processing
	frame = cv2.resize(frame, (640, 480))
	iq: ImageQuery = gl.submit_image_query(detector=detector, image=frame)
	print("count: ", iq.result.count)
	print("confidence: ", iq.result.confidence)		