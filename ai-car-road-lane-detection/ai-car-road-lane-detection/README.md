# AI Car: Real-Time Road Damage & Lane Detection

A computer-vision pipeline that simulates part of an autonomous/driver-assist
vehicle's perception stack: it detects **lane markings** and flags likely
**potholes / road-surface damage** in a video feed, frame by frame, and
overlays both onto the output for a simple heads-up display.

Built as a portfolio/coursework project to demonstrate classical image
processing (OpenCV) and a lightweight, extensible detection architecture —
not a production ADAS system. See [Limitations](#limitations--honest-scope)
below for what it does and doesn't handle.

## Demo

No dashcam footage is bundled with this repo, so a small synthetic road
scene is generated on first run (`sample_data/sample_road.jpg` /
`sample_road.mp4`) — flat gray road, two lane lines converging to a vanishing
point, a dashed center line, and a few dark irregular blobs standing in for
potholes. Running the pipeline on it produces something like:

- Lane lines detected and filled in translucent green between the boundaries
- Damage regions boxed in red with a confidence score, e.g. `pothole 0.85`
- A small HUD strip at the top showing live FPS, lane status, and damage count

Swap in real dashcam video any time — see [Testing with your own
footage](#testing-with-your-own-footage).

## Features

- **Lane detection** using a classic Canny edge → ROI mask → probabilistic
  Hough transform → slope-based left/right line averaging pipeline, with
  frame-to-frame smoothing so the drawn lane doesn't jitter.
- **Pothole / road-damage detection** using an adaptive-threshold +
  contour-shape heuristic (no training data required to run), with an
  optional pluggable ML classifier slot for a trained model once labeled
  data is available.
- **Frame-by-frame video processing** (`main.py`) that runs both pipelines
  on a video file, a webcam stream, or a single image, drawing all overlays
  and a live FPS/status HUD, and writing the annotated output to disk.
- **Synthetic sample data generator** so the whole pipeline is runnable
  immediately with zero external datasets.
- **Non-max suppression** on damage detections to avoid duplicate/overlapping
  boxes around the same region.

## Tech Stack

- **Python 3.9+**
- **OpenCV** (`opencv-python-headless`) — image processing, edge/contour
  detection, Hough transform, drawing, video I/O
- **NumPy** — array/matrix operations underlying the CV pipeline
- **Pillow** — used by the sample-data generator
- Optional: **scikit-learn** + **joblib** (only if you train the optional
  pothole classifier — see [`src/train_classifier.py`](src/train_classifier.py))

## Project Structure

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
├── sample_data/                # Generated synthetic image/video (gitignored input,
│                                #   regenerate any time — see below)
├── models/                     # (optional) trained pothole_classifier.joblib goes here
└── output/                     # Annotated results are written here
```

## Setup / Installation

```bash
git clone <this-repo-url>
cd ai-car-road-lane-detection

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**1. Generate the synthetic sample data** (safe to skip — `main.py` will use
whatever is already in `sample_data/`, but you need to create it once):

```bash
python src/generate_sample_data.py
```

This writes `sample_data/sample_road.jpg` and `sample_data/sample_road.mp4`.

**2. Run the pipeline on the sample video:**

```bash
python src/main.py --source sample_data/sample_road.mp4 --output output/annotated_result.mp4
```

**3. Run on a single image:**

```bash
python src/main.py --source sample_data/sample_road.jpg --image --output output/annotated_result.jpg
```

**4. Run live from a webcam** (device 0), showing a live window:

```bash
python src/main.py --source 0 --display
```

`--display` requires a GUI-capable OpenCV build (`pip install opencv-python`
instead of the headless package) and an environment with a display. Without
it, `main.py` runs fully headless and just writes the annotated file — handy
for servers/CI/cloud notebooks.

### CLI options (`src/main.py`)

| Flag | Description |
|---|---|
| `--source` | Video file path, image path, or webcam index (default: bundled sample video) |
| `--image` | Treat `--source` as a single still image |
| `--output` | Where to write the annotated result (default `output/annotated_result.mp4` / `.jpg`) |
| `--display` | Also open a live `cv2.imshow` window |
| `--max-frames` | Cap the number of frames processed (useful for a quick smoke test) |

### Testing with your own footage

Point `--source` at any dashcam/phone video of a road:

```bash
python src/main.py --source /path/to/your_drive.mp4 --output output/your_drive_annotated.mp4
```

Two things to check first if results look off:

- **ROI geometry** — `default_roi_vertices()` in `utils.py` and
  `_road_roi_mask()` in `pothole_detection.py` assume a roughly centered,
  forward-facing camera with the horizon around 55–58% down the frame. If
  your camera is mounted differently, adjust the fractional coordinates
  there.
- **Threshold tuning** — `LaneDetector(canny_low=..., canny_high=...,
  hough_threshold=...)` and `PotholeDetector(block_size=..., c=...,
  min_confidence=...)` expose the key parameters; real footage (varying
  lighting, asphalt color, shadows) will likely need some retuning from the
  defaults used for the clean synthetic sample.

## How It Works

### Lane detection pipeline (`lane_detection.py`)

1. **Grayscale + Gaussian blur** — reduces color/texture noise before edge
   detection.
2. **Canny edge detection** — finds high-gradient pixels (line boundaries).
3. **Region-of-interest masking** — restricts edges to a trapezoid over the
   road surface, discarding sky/roadside clutter that would otherwise
   produce false line matches.
4. **Probabilistic Hough transform** (`cv2.HoughLinesP`) — finds straight
   line segments among the masked edge pixels.
5. **Slope-based grouping** — segments are split into "left lane" (negative
   slope) and "right lane" (positive slope) groups, filtering out
   near-horizontal noise, then each group's slope/intercept is averaged into
   one line.
6. **Temporal smoothing** — each frame's averaged slope/intercept is
   blended with a short rolling history so the drawn line doesn't jump
   around when a frame briefly has few good Hough matches.
7. **Overlay** — the two extrapolated lines (and the filled area between
   them, when both are found) are drawn back onto the frame.

This is a **straight-line model**: it works well on relatively straight
road segments with visible markings and will under-fit tight curves — a
known, documented limitation of this classic approach (a production system
would fit a polynomial/spline or use a learned lane model instead).

### Pothole / road-damage detection (`pothole_detection.py`)

No labeled pothole dataset was available for this project, so the module is
built in two layers so it's honest about what's learned vs. rule-based, and
easy to extend later:

1. **Heuristic detector (default, no training data needed):**
   - Restrict to a wide road-surface ROI.
   - Blur + **adaptive thresholding** to isolate blobs that are locally
     darker than their surroundings (the shadow/depth cue inside a pothole).
   - Morphological open/close to remove speckle noise and merge fragments
     of the same damage region.
   - Extract contours and compute shape features per candidate: area
     relative to frame, aspect ratio, extent (fill ratio vs. bounding box),
     solidity (vs. convex hull), and Laplacian variance (a texture-roughness
     proxy).
   - Hard shape gates reject candidates that are geometrically implausible
     as potholes (too large/elongated a bounding box, too little fill, too
     needle-shaped) — this specifically prevents lane-marking edges and
     dashed center-line segments from being mistaken for damage.
   - Remaining candidates are scored 0–1 against typical pothole shape
     ranges; only ones above a confidence threshold are kept, then
     **non-max suppression** removes duplicate overlapping boxes.

2. **Optional pluggable ML classifier:** `PotholeClassifier` extracts the
   same shape feature vector and, if a trained scikit-learn model file
   exists at `models/pothole_classifier.joblib`, uses it to re-score
   candidates instead of the fixed heuristic rule. Without a model file
   present (the default), it transparently falls back to the heuristic
   scorer, so the pipeline always runs out of the box. `train_classifier.py`
   is a template for training that model once you have labeled patch data
   (e.g. from a public pothole dataset or your own labeled crops).

This mirrors how many real "pothole detection" prototypes are structured:
classical CV for candidate generation, with a slot left open for a learned
classifier once labeled data is collected.

### Bringing it together (`main.py`)

For each frame: resize → run lane detection → draw lane overlay → run
pothole detection → draw damage boxes → draw the FPS/status HUD → write/
display. Running frame-by-frame over a video file this way simulates how
the same pipeline would run on a live camera feed in a real-time system.

## Limitations & Honest Scope

This is a student/portfolio project demonstrating the CV techniques, not a
validated safety system:

- Lane model is a straight line per side — struggles on sharp curves,
  faded/missing markings, or heavy occlusion (traffic, shadows).
- Pothole detection is a **shape heuristic**, not a trained object
  detector — it can miss subtle or low-contrast damage, and misclassify
  other locally-dark, blob-shaped regions (oil stains, manhole covers,
  debris) as damage. Its accuracy has not been benchmarked against a labeled
  real-world dataset.
- ROI geometry assumes a forward-facing, roughly centered camera and needs
  re-tuning for different mounting positions.
- Frame rate figures printed by `main.py` reflect this machine's CPU-only
  OpenCV performance on the given input resolution — they are not a claim
  about performance on embedded/automotive hardware.

## Possible Extensions

- Fit a polynomial (instead of a straight line) for curved-lane support.
- Train the optional `PotholeClassifier` on a real labeled pothole dataset.
- Add a lightweight CNN (e.g. a small Keras model) as a second-stage
  verifier on top of the heuristic candidates, instead of hand-crafted
  shape features.
- Lane-departure and time-to-collision style alerts built on top of the
  existing lane/damage outputs.

## License

MIT — feel free to reuse for learning purposes.
