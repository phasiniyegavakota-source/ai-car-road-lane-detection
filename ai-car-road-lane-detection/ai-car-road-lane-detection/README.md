🚗 AI Car — Real-Time Road Damage & Lane Detection
===================================================

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

A computer-vision pipeline that simulates part of an autonomous/driver-assist
vehicle's perception stack: it detects **lane markings** and flags likely
**potholes / road-surface damage** in a video feed, frame by frame, and
overlays both onto the output for a simple heads-up display.

Built as a portfolio/coursework project to demonstrate classical image
processing (OpenCV) and a lightweight, extensible detection architecture —
not a production ADAS system. See [Limitations](#limitations--honest-scope)
for what it does and doesn't handle.

## Tech Stack

- **Language:** Python 3.9+
- **Computer Vision:** OpenCV (`opencv-python-headless`) — edge detection, Hough
  transform, contour analysis, video I/O
- **Numerical Computing:** NumPy
- **Sample Data:** Pillow (synthetic road-scene generator)
- **Optional ML:** scikit-learn + joblib (for the pluggable pothole classifier)

## Features

- 🛣️ **Lane Detection** — Canny edge → ROI mask → probabilistic Hough
  transform → slope-based left/right line averaging, with frame-to-frame
  smoothing so the drawn lane doesn't jitter
- 🕳️ **Pothole / Road-Damage Detection** — adaptive-threshold + contour-shape
  heuristic (no training data required to run), with a pluggable ML
  classifier slot for a trained model
- 🎥 **Real-Time Video Pipeline** — processes a video file, webcam stream, or
  single image frame-by-frame, drawing overlays and a live FPS/status HUD
- 🧪 **Synthetic Sample Data Generator** — the whole pipeline runs immediately
  with zero external datasets required
- ✅ **Non-Max Suppression** — removes duplicate/overlapping detection boxes
  around the same damage region

## Application Structure

```
ai-car-road-lane-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── lane_detection.py       # Classic CV lane-finding pipeline
│   ├── pothole_detection.py    # Heuristic + pluggable-ML damage detector
│   ├── main.py                 # Ties both pipelines together, video/image I/O
│   ├── utils.py                # Shared helpers (resize, ROI mask, HUD, labels)
│   ├── generate_sample_data.py # Synthesizes a test image + video
│   └── train_classifier.py     # Template for training the optional ML classifier
├── sample_data/                # Generated synthetic image/video
├── models/                     # (optional) trained pothole_classifier.joblib
└── output/                     # Annotated results written here
```

## How to Run

```bash
git clone <this-repo-url>
cd ai-car-road-lane-detection

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Generate the synthetic sample road image/video (one-time)
python src/generate_sample_data.py

# Run the pipeline on the sample video
python src/main.py --source sample_data/sample_road.mp4 --output output/annotated_result.mp4

# Or run on a single image
python src/main.py --source sample_data/sample_road.jpg --image --output output/annotated_result.jpg

# Or run live from a webcam, with a display window
python src/main.py --source 0 --display
```

Open `output/annotated_result.mp4` (or `.jpg`) to see the lane overlay and
any flagged damage regions.

### CLI options (`src/main.py`)

| Flag | Description |
|---|---|
| `--source` | Video file path, image path, or webcam index (default: bundled sample video) |
| `--image` | Treat `--source` as a single still image |
| `--output` | Where to write the annotated result |
| `--display` | Also open a live `cv2.imshow` window |
| `--max-frames` | Cap the number of frames processed (quick smoke test) |

### Testing with your own footage

```bash
python src/main.py --source /path/to/your_drive.mp4 --output output/your_drive_annotated.mp4
```

If results look off on real footage, check **ROI geometry** (assumes a
roughly centered, forward-facing camera — adjust `default_roi_vertices()` in
`utils.py` if yours differs) and **threshold tuning** (`LaneDetector` /
`PotholeDetector` constructor parameters — real-world lighting and asphalt
color will likely need some retuning from the clean synthetic defaults).

## How It Works

**Lane detection:** grayscale + Gaussian blur → Canny edge detection →
region-of-interest masking (trapezoid over the road surface) → probabilistic
Hough transform (`cv2.HoughLinesP`) to find line segments → slope-based
grouping into left/right lane candidates → temporal smoothing across frames
→ overlay. This is a straight-line model, so it works well on relatively
straight road segments and under-fits tight curves — a known limitation of
the classical approach.

**Pothole detection:** restrict to a road-surface ROI → adaptive thresholding
to isolate locally-darker blobs (the shadow/depth cue inside a pothole) →
morphological cleanup → contour shape features (area, aspect ratio, extent,
solidity, Laplacian variance/texture) → hard shape gates reject
geometrically-implausible candidates (this is what stops lane markings from
being mistaken for damage) → confidence scoring → non-max suppression. A
`PotholeClassifier` slot exists to swap in a trained scikit-learn model in
place of the heuristic once labeled data is available (`train_classifier.py`
is the training template).

## Results / Outcomes

- ✅ Implemented a full classical-CV lane detection pipeline (Canny + Hough
  transform) with frame-to-frame smoothing
- ✅ Built a heuristic pothole/damage detector using contour shape analysis,
  with hard gates that prevent lane-marking false positives
- ✅ Designed a pluggable architecture — the same detector interface works
  with either the built-in heuristic or a trained ML classifier
- ✅ Runs end-to-end out of the box via a synthetic sample-data generator, no
  external dataset required
- ✅ Verified against a generated test image/video with zero crashes across
  the full video/image/webcam code paths

## Limitations & Honest Scope

This is a student/portfolio project demonstrating the CV techniques, not a
validated safety system:

- Lane model is a straight line per side — struggles on sharp curves, faded
  markings, or heavy occlusion
- Pothole detection is a shape heuristic, not a trained object detector —
  accuracy has not been benchmarked against a labeled real-world dataset
- ROI geometry assumes a forward-facing, centered camera and needs retuning
  for different mounts
- Reported FPS reflects CPU-only performance on this machine, not a claim
  about embedded/automotive hardware

## Possible Extensions

- Fit a polynomial (instead of a straight line) for curved-lane support
- Train the optional `PotholeClassifier` on a real labeled pothole dataset
- Add a lightweight CNN as a second-stage verifier over the heuristic candidates
- Lane-departure / time-to-collision alerts built on the existing outputs

---

Developer: **Hasini Prasad** | [LinkedIn](https://linkedin.com/in/hasini-prasad-b761b0286)
