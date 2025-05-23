# Employee Counting with Groundlight

This repository uses the [Groundlight](https://groundlight.ai) computer vision platform to count the number of employees entering and exiting through a doorway. <i>(Note: you will need at least a Business-level account to access Groundlight's Counting mode - email support@groundlight.ai if you're interested in a free one month trial.)</i>

<p align="center">
<img src="results/Final_Counting_demo_with_dashboard.webp" alt="Demo GIF" width="640" />
</p>

<p align="center">
  <a href="https://vimeo.com/1086127262" target="_blank">
    Watch in full resolution on Vimeo
  </a>
</p>

## Quick Overview

- A sample video is provided in the `./video/` folder.
- `fetch_frames.py` extracts individual frames from the video and saves them to `./video/frames/`.
- `create_detector_submit_images.py` samples every 10th frame from the **first half** of the video and submits them to Groundlight to train a custom detector.
- `enter_exit_count.py` processes the full video (sampling every 3rd frame for speed), sends the frames to the trained Groundlight detector, and uses an IoU-based tracker to track people wearing blue Groundlight t-shirts entering and exiting the door.
- A results video with metadata overlay is saved back to the `./video/` folder.

---

## Step-by-Step Developer Guide

### Step 1: Set Up Your Environment

Before getting started, ensure you have **Python 3.9 or higher** installed. This is a **Poetry-managed** project, which handles dependencies and virtual environments.

#### Clone the Repository
```bash
git clone https://github.com/groundlight/employee-counting.git
cd employee-counting
```

#### Install Poetry (if not already installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Install Project Dependencies
```bash
poetry install
```

#### Obtain a Groundlight API Token
Sign up at [dashboard.groundlight.ai](https://dashboard.groundlight.ai) and retrieve your API token. Then export it:

```bash
export GROUNDLIGHT_API_TOKEN='your-api-token'
```
<img src="https://cdn.prod.website-files.com/664b7cc2ac49aeb2da6ef0f4/6823a63169bbb9bdc3eff8a5_APItoken.png" alt="API Token Screenshot" width="60%" />

---

### Step 2: Prepare Your Video Data and Parse Frames

Place your surveillance video (e.g., from a static camera) inside the `./video/` directory.

Extract frames:
```bash
poetry run python fetch_frames.py
```

> This saves individual frames to `./video/frames/` for later use in detector training and tracking.

---

### Step 3: Create and Train a Detector

Create a detector and submit sample frames:
```bash
poetry run python create_detector_submit_images.py
```

> This script samples every 10th frame from the first half of the video and submits them to **Groundlight** for training a custom detector focused on identifying employees (e.g., wearing blue shirts).  
> Groundlight handles **labeling, training, and hosting** — no ML expertise required!

---

### Step 4: Run the Tracker to Count Entries/Exits

Detect and track people moving through the doorway:
```bash
poetry run python enter_exit_counts.py
```

> Uses IoU-based tracking and the trained Groundlight detector to count individuals crossing a virtual line, sampling every 3rd frame for efficiency.

---

### Step 5: Review the Output and Visualize Results

After processing, results are saved as an **annotated video** in the `./video/` directory, including:

- Bounding boxes around detected employees  
- Entry and exit counts  
- Timestamps for each event  

This helps validate detections and assess system accuracy.

---

## Additional Resources

- [**Groundlight Python SDK Documentation**](https://code.groundlight.ai/python-sdk/docs/getting-started): Get started with Groundlight's Python SDK
- [**Counting Detectors Guide**](https://code.groundlight.ai/python-sdk/docs/answer-modes/counting-detectors): Get started with Groundlight's Counting Mode

## License

This project is licensed under the [Apache License 2.0](LICENSE).

