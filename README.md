# Employee Counting with Groundlight

This repository uses the [Groundlight](https://groundlight.ai) service to count the number of employees entering and exiting through a doorway.

## Overview

- A sample video is provided in the `./video/` folder.
- `fetch_frames.py` extracts individual frames from the video and saves them to `./video/frames/`.
- `create_detector_submit_images.py` samples every 10th frame from the **first half** of the video and submits them to Groundlight to train a custom detector.
- `enter_exit_count.py` processes the full video (sampling every 3rd frame for speed), sends the frames to the trained Groundlight detector, and uses an IoU-based tracker to track people wearing blue Groundlight t-shirts entering and exiting the door.
- A results video with metadata overlay is saved back to the `./video/` folder.