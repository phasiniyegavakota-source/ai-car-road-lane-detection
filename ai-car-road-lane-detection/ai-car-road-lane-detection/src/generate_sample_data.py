"""
generate_sample_data.py
------------------------
No real dashcam footage is bundled with this repo (for size and licensing
reasons), so this script synthesizes a small test image and a short test
video of a simple road scene: gray road surface, two white lane lines
converging toward a vanishing point, and a few dark irregular blobs standing
in for potholes. It gives the pipeline something concrete to run against out
of the box; swap in your own dashcam footage for real testing (see README).

Run directly:
    python src/generate_sample_data.py

Produces:
    sample_data/sample_road.jpg
    sample_data/sample_road.mp4   (a few seconds, camera drifting slightly
                                    frame to frame so lane smoothing has
                                    something to do)
"""

import os
import random
import numpy as np
import cv2


WIDTH, HEIGHT = 960, 540
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sample_data")


def _draw_road_scene(offset: int = 0, seed: int = 0) -> np.ndarray:
    """Render one synthetic road frame.

    `offset` shifts the lane geometry slightly left/right frame to frame to
    simulate a car drifting within its lane, which is enough to make the
    smoothing in LaneDetector visibly do something across a short clip.
    """
    rng = random.Random(seed)
    img = np.full((HEIGHT, WIDTH, 3), (60, 60, 60), dtype=np.uint8)  # asphalt gray

    # Sky
    cv2.rectangle(img, (0, 0), (WIDTH, int(HEIGHT * 0.42)), (150, 120, 90), -1)
    # Horizon haze band
    cv2.rectangle(img, (0, int(HEIGHT * 0.38)), (WIDTH, int(HEIGHT * 0.44)),
                  (140, 130, 120), -1)

    horizon_y = int(HEIGHT * 0.55)
    bottom_y = HEIGHT

    center_x = WIDTH // 2 + offset
    lane_half_width_top = 40
    lane_half_width_bottom = 340

    left_top = (center_x - lane_half_width_top, horizon_y)
    right_top = (center_x + lane_half_width_top, horizon_y)
    left_bottom = (center_x - lane_half_width_bottom, bottom_y)
    right_bottom = (center_x + lane_half_width_bottom, bottom_y)

    # Slightly darker road wedge between the lane lines vs. shoulders
    road_poly = np.array([left_top, right_top, right_bottom, left_bottom], np.int32)
    cv2.fillPoly(img, [road_poly], (70, 70, 70))

    # Lane lines (solid white, with a little noise/anti-aliasing via blur later)
    cv2.line(img, left_top, left_bottom, (235, 235, 235), 8, cv2.LINE_AA)
    cv2.line(img, right_top, right_bottom, (235, 235, 235), 8, cv2.LINE_AA)

    # Dashed center line
    n_dashes = 8
    for i in range(n_dashes):
        t1 = i / n_dashes
        t2 = (i + 0.5) / n_dashes
        x1 = int(center_x + (0) * (1 - t1))
        y1 = int(horizon_y + (bottom_y - horizon_y) * t1)
        y2 = int(horizon_y + (bottom_y - horizon_y) * t2)
        cv2.line(img, (center_x, y1), (center_x, y2), (200, 200, 200), 4, cv2.LINE_AA)

    # A few pothole-like dark irregular blobs on the road surface. Positions
    # are hand-picked to sit clearly inside the lane (clear of the lane
    # lines and the dashed center line, and clear of each other) so the
    # heuristic contour detector -- which relies on finding an isolated,
    # roughly convex dark blob -- has a fair, unambiguous target. Potholes
    # that visually touch a lane marking would get merged with it during
    # the morphological closing step, which is a real limitation of this
    # simple contour-based approach (see README "Limitations").
    pothole_specs = [
        (360, 420, 32, 20),
        (640, 460, 28, 18),
        (420, 500, 42, 24),
    ]
    for (px, py, rw, rh) in pothole_specs:
        px += offset
        pts = []
        n_pts = 10
        for k in range(n_pts):
            angle = 2 * np.pi * k / n_pts
            jitter = 0.7 + 0.3 * rng.random()
            x = px + int(rw * jitter * np.cos(angle))
            y = py + int(rh * jitter * np.sin(angle))
            pts.append([x, y])
        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(img, pts, (25, 25, 25))
        cv2.polylines(img, pts, True, (10, 10, 10), 2, cv2.LINE_AA)

    # Mild blur + noise so it isn't perfectly clean synthetic edges
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.default_rng(seed).normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_image(path: str) -> None:
    frame = _draw_road_scene(offset=0, seed=42)
    cv2.imwrite(path, frame)
    print(f"Wrote sample image: {path}")


def generate_video(path: str, seconds: int = 4, fps: int = 20) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (WIDTH, HEIGHT))
    n_frames = seconds * fps
    for i in range(n_frames):
        # Gentle sinusoidal drift to simulate steering wander
        offset = int(25 * np.sin(2 * np.pi * i / n_frames))
        frame = _draw_road_scene(offset=offset, seed=42 + i)
        writer.write(frame)
    writer.release()
    print(f"Wrote sample video: {path} ({n_frames} frames @ {fps}fps)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    generate_image(os.path.join(OUT_DIR, "sample_road.jpg"))
    generate_video(os.path.join(OUT_DIR, "sample_road.mp4"))


if __name__ == "__main__":
    main()
