"""
main.py
-------
Entry point that ties lane detection and pothole detection together and
runs them frame-by-frame over a video source (a video file, a webcam index,
or a single still image), drawing lane overlays and damage-region boxes and
writing/showing the annotated result. This is the "real-time simulation"
described in the project write-up: each frame is processed independently
through both pipelines and annotated with a small on-screen HUD.

Usage
=====
    # Run on the bundled synthetic sample video
    python src/main.py --source sample_data/sample_road.mp4

    # Run on a single image
    python src/main.py --source sample_data/sample_road.jpg --image

    # Run on your own dashcam footage
    python src/main.py --source /path/to/your_dashcam.mp4 --output output/result.mp4

    # Run live from a webcam (device 0)
    python src/main.py --source 0 --display

No GUI/display is required to test the pipeline -- by default the script
writes an annotated video/image to output/ so it works headlessly (useful in
CI / cloud environments without a display). Pass --display to also open a
live cv2.imshow window when running somewhere with a display available.
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lane_detection import LaneDetector
from pothole_detection import PotholeDetector
from utils import resize_frame, stack_hud


def process_frame(frame, lane_detector: LaneDetector, pothole_detector: PotholeDetector):
    frame = resize_frame(frame, width=960)

    left_line, right_line = lane_detector.detect(frame)
    annotated = lane_detector.draw(frame, left_line, right_line)

    detections = pothole_detector.detect(frame)
    annotated = pothole_detector.draw(annotated, detections)

    lane_status = lane_detector.status(left_line, right_line)
    return annotated, lane_status, len(detections)


def run_on_image(source_path: str, output_path: str, display: bool):
    frame = cv2.imread(source_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    lane_detector = LaneDetector()
    pothole_detector = PotholeDetector()

    start = time.time()
    annotated, lane_status, n_potholes = process_frame(frame, lane_detector, pothole_detector)
    elapsed = time.time() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    annotated = stack_hud(annotated, fps, lane_status, n_potholes)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"[main] Processed image in {elapsed*1000:.1f} ms")
    print(f"[main] Lane status: {lane_status} | Damage regions: {n_potholes}")
    print(f"[main] Wrote annotated image to: {output_path}")

    if display:
        cv2.imshow("AI Car - Lane & Pothole Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(source, output_path: str, display: bool, max_frames: int = None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    lane_detector = LaneDetector()
    pothole_detector = PotholeDetector()

    writer = None
    frame_count = 0
    total_start = time.time()
    fps_smoothed = 0.0

    print(f"[main] Processing video source: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        annotated, lane_status, n_potholes = process_frame(frame, lane_detector, pothole_detector)
        dt = time.time() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smoothed = inst_fps if fps_smoothed == 0 else 0.9 * fps_smoothed + 0.1 * inst_fps
        annotated = stack_hud(annotated, fps_smoothed, lane_status, n_potholes)

        if writer is None and output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

        if writer is not None:
            writer.write(annotated)

        if display:
            cv2.imshow("AI Car - Lane & Pothole Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - total_start
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
    print(f"[main] Processed {frame_count} frames in {total_elapsed:.2f}s "
          f"(avg {avg_fps:.1f} fps on this machine, CPU-only)")
    if output_path and writer is not None:
        print(f"[main] Wrote annotated video to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time lane detection + pothole/road-damage detection.")
    parser.add_argument("--source", default="sample_data/sample_road.mp4",
                         help="Path to a video file, path to an image, or a webcam "
                              "index (e.g. 0). Defaults to the bundled synthetic sample.")
    parser.add_argument("--image", action="store_true",
                         help="Treat --source as a single still image instead of video.")
    parser.add_argument("--output", default=None,
                         help="Where to write the annotated result. Defaults to "
                              "output/annotated_result.mp4 (or .jpg for --image).")
    parser.add_argument("--display", action="store_true",
                         help="Also show a live cv2.imshow window (requires a display).")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Optional cap on number of frames processed (useful for quick tests).")
    return parser.parse_args()


def main():
    args = parse_args()

    is_image = args.image or str(args.source).lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        output_path = args.output or "output/annotated_result.jpg"
        run_on_image(args.source, output_path, args.display)
    else:
        output_path = args.output or "output/annotated_result.mp4"
        # Allow numeric webcam indices passed as strings, e.g. "0"
        source = args.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        run_on_video(source, output_path, args.display, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
