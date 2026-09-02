"""
lane_detection.py
------------------
Classic (non-deep-learning) lane detection pipeline:

    1. Grayscale conversion
    2. Gaussian blur (noise suppression)
    3. Canny edge detection
    4. Region-of-interest (ROI) masking to the road area
    5. Probabilistic Hough transform to find candidate line segments
    6. Slope-based filtering + averaging into a single left/right lane line
    7. Overlay of the extrapolated lane lines back onto the original frame

This mirrors the standard approach used in most introductory self-driving-car
lane-finding projects. It is deliberately dependency-light (OpenCV + NumPy
only) so it runs in real time on CPU and needs no trained model or GPU.

Limitations (documented honestly, as this is a student project, not a
production system):
    - Assumes reasonably well-marked lanes and a forward-facing, roughly
      centered camera.
    - Straight-line (linear) lane model -- it will not fit tight curves well.
    - Sensitive to the Canny/Hough thresholds and the ROI polygon, which may
      need re-tuning for different camera mounts/resolutions/lighting.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

from utils import region_of_interest, default_roi_vertices


@dataclass
class LaneLine:
    """A single averaged lane line expressed as two endpoints."""
    x1: int
    y1: int
    x2: int
    y2: int
    slope: float
    intercept: float


class LaneDetector:
    """Stateful lane detector.

    Keeping a short history of recently-detected left/right slopes lets us
    smooth the drawn line across frames so it doesn't visibly jitter when a
    frame briefly has few/no Hough line matches (e.g. a faded lane marking).
    """

    def __init__(self,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 hough_threshold: int = 20,
                 min_line_length: int = 20,
                 max_line_gap: int = 300,
                 smoothing_window: int = 8):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.smoothing_window = smoothing_window

        self._left_history: List[Tuple[float, float]] = []
        self._right_history: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges

    def masked_edges(self, edges: np.ndarray) -> np.ndarray:
        h, w = edges.shape[:2]
        vertices = default_roi_vertices(w, h)
        return region_of_interest(edges, vertices)

    def hough_lines(self, masked_edges: np.ndarray) -> Optional[np.ndarray]:
        return cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

    # ------------------------------------------------------------------
    # Line averaging / extrapolation
    # ------------------------------------------------------------------
    def _average_slope_intercept(self, lines: Optional[np.ndarray]
                                  ) -> Tuple[Optional[Tuple[float, float]],
                                             Optional[Tuple[float, float]]]:
        """Split raw Hough segments into left/right groups by slope sign and
        average each group into a single (slope, intercept) pair.

        Slope < 0 (in image coordinates, y grows downward) -> left lane,
        slope > 0 -> right lane. Near-horizontal segments are discarded as
        noise (they are almost never real lane markings).
        """
        left_fits = []
        right_fits = []
        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # vertical line, undefined slope
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.3:
                continue  # too horizontal, likely not a lane marking
            if slope < 0:
                left_fits.append((slope, intercept))
            else:
                right_fits.append((slope, intercept))

        left_avg = np.mean(left_fits, axis=0) if left_fits else None
        right_avg = np.mean(right_fits, axis=0) if right_fits else None
        return (tuple(left_avg) if left_avg is not None else None,
                tuple(right_avg) if right_avg is not None else None)

    def _smooth(self, history: List[Tuple[float, float]],
                new_value: Optional[Tuple[float, float]]
                ) -> Optional[Tuple[float, float]]:
        if new_value is not None:
            history.append(new_value)
        if len(history) > self.smoothing_window:
            history.pop(0)
        if not history:
            return None
        arr = np.array(history)
        return tuple(arr.mean(axis=0))

    def _make_line_points(self, slope_intercept: Optional[Tuple[float, float]],
                           y1: int, y2: int) -> Optional[LaneLine]:
        if slope_intercept is None:
            return None
        slope, intercept = slope_intercept
        if abs(slope) < 1e-6:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return LaneLine(x1=x1, y1=y1, x2=x2, y2=y2, slope=slope, intercept=intercept)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Tuple[Optional[LaneLine], Optional[LaneLine]]:
        """Run the full pipeline on one BGR frame and return the smoothed
        (left_lane, right_lane) LaneLine objects (either may be None if not
        confidently detected in this frame)."""
        edges = self.preprocess(frame)
        masked = self.masked_edges(edges)
        raw_lines = self.hough_lines(masked)
        left_fit, right_fit = self._average_slope_intercept(raw_lines)

        left_fit = self._smooth(self._left_history, left_fit)
        right_fit = self._smooth(self._right_history, right_fit)

        h = frame.shape[0]
        y1 = h
        y2 = int(h * 0.6)
        left_line = self._make_line_points(left_fit, y1, y2)
        right_line = self._make_line_points(right_fit, y1, y2)
        return left_line, right_line

    def draw(self, frame: np.ndarray,
              left_line: Optional[LaneLine],
              right_line: Optional[LaneLine],
              line_color: Tuple[int, int, int] = (0, 255, 0),
              line_thickness: int = 8,
              fill_lane: bool = True) -> np.ndarray:
        """Overlay the detected lane lines (and, if both are present, a
        translucent fill of the lane area between them) onto a copy of the
        input frame."""
        overlay = np.zeros_like(frame)

        if left_line is not None:
            cv2.line(overlay, (left_line.x1, left_line.y1),
                      (left_line.x2, left_line.y2), line_color, line_thickness)
        if right_line is not None:
            cv2.line(overlay, (right_line.x1, right_line.y1),
                      (right_line.x2, right_line.y2), line_color, line_thickness)

        if fill_lane and left_line is not None and right_line is not None:
            pts = np.array([
                [left_line.x1, left_line.y1],
                [left_line.x2, left_line.y2],
                [right_line.x2, right_line.y2],
                [right_line.x1, right_line.y1],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 80, 0))

        return cv2.addWeighted(frame, 1.0, overlay, 0.6, 0)

    def status(self, left_line: Optional[LaneLine],
               right_line: Optional[LaneLine]) -> str:
        if left_line and right_line:
            return "OK"
        if left_line or right_line:
            return "PARTIAL"
        return "NOT FOUND"
