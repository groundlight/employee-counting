from groundlight import ExperimentalApi, Detector, ImageQuery
from fetch_frames import FRAMES_DIR
from glob import glob
import cv2
import argparse

SAMPLE_INTERVAL = 10 # sample every 10th frame

def create_detector_submit_images(detector_name, detector_query, class_name):
	gl = ExperimentalApi()

	# TODO: update to use get_or_create_detector once mode can be specified
	try:
		detector: Detector = gl.get_detector_by_name(detector_name)
	except:
		detector: Detector = gl.create_counting_detector(
			name=detector_name,
			query=detector_query,
			class_name=class_name,
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create detector and submit images for analysis")
	
	parser.add_argument("--name", "-n",
						default="employees_with_groundlight_tshirt",
						help="Name of the detector (default: employees_with_groundlight_tshirt)")
	parser.add_argument("--query", "-q",
						default="Label each employee with Groundlight T-shirt in the image", 
						help="Query for the detector (default: Label each employee with Groundlight T-shirt in the image)")
	parser.add_argument("--class-name", "-c",
						default="employee",
						help="Name of the class to count (default: employee)")
	
	args = parser.parse_args()
	create_detector_submit_images(args.name, args.query, args.class_name)