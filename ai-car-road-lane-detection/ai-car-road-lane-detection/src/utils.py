"""
utils.py
--------
Small shared helpers used by both the lane detection and pothole detection
modules: frame resizing, region-of-interest masking, and simple drawing
utilities. Keeping these in one place avoids duplicating boilerplate OpenCV
code across modules.
"""

from typing import Tuple, List
import numpy as np
import cv2


def resize_frame(frame: np.ndarray, width: int = 960) -> np.ndarray:
    """Resize a frame to a fixed width while preserving aspect ratio.

    Real-time CV pipelines are typically run on a downscaled frame to keep
    per-frame processing time low; the caller can upscale the final overlay
    if needed.
    """
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def region_of_interest(img: np.ndarray, vertices: List[np.ndarray]) -> np.ndarray:
    """Mask out everything outside the given polygon.

    For a forward-facing dashcam, the road surface roughly occupies a
    trapezoid in the lower half of the frame. Masking to that region before
    edge/line detection removes buildings, sky, and other clutter that would
    otherwise generate false-positive lines.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


def default_roi_vertices(width: int, height: int) -> List[np.ndarray]:
    """A reasonable default trapezoidal ROI for a forward-facing camera.

    Bottom corners sit at the edges of the frame; top corners pinch inward
    toward the vanishing point roughly 55% down the image. Tuned for a
    typical dashcam mounting height/angle -- adjust for your own footage.
    """
    bottom_left = (int(width * 0.05), height)
    top_left = (int(width * 0.45), int(height * 0.58))
    top_right = (int(width * 0.55), int(height * 0.58))
    bottom_right = (int(width * 0.95), height)
    return [np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)]


def put_label(img: np.ndarray, text: str, org: Tuple[int, int],
              color: Tuple[int, int, int] = (0, 255, 255),
              scale: float = 0.6, thickness: int = 2) -> None:
    """Draw a small text label with a dark outline so it stays readable
    against both light and dark road backgrounds."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def stack_hud(frame: np.ndarray, fps: float, lane_status: str,
              pothole_count: int) -> np.ndarray:
    """Draw a small heads-up-display strip at the top of the frame showing
    the current frame rate and detection summary."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (20, 20, 20), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    put_label(frame, f"FPS: {fps:5.1f}", (10, 23), color=(0, 255, 0), scale=0.55)
    put_label(frame, f"Lane: {lane_status}", (150, 23), color=(0, 255, 255), scale=0.55)
    put_label(frame, f"Damage regions: {pothole_count}", (340, 23),
              color=(0, 140, 255), scale=0.55)
    return frame
