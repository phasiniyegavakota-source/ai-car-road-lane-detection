"""
pothole_detection.py
---------------------
Road-surface damage (pothole) detection.

Architecture
============
Real pothole datasets (labeled dashcam/drone imagery) are not available in
this environment, so this module is built in two layers so it is honest
about what is trained vs. heuristic, while still being easy to extend with
real data later:

1. **Heuristic detector (default, runs out of the box, no training data
   needed).** Potholes tend to show up as irregular, locally-darker patches
   on an otherwise fairly uniform road surface, often with a visible rim/
   shadow edge. The heuristic pipeline:

        - Restrict to a road-surface ROI (reuses the same trapezoid idea as
          lane detection, just wider/shorter).
        - Grayscale + blur, then adaptive thresholding to pull out blobs
          that are darker than their local neighborhood (shadows inside a
          pothole are usually a strong cue).
        - Morphological close/open to remove speckle noise and merge
          fragments of the same damage region.
        - Contour extraction, then filter contours by area, aspect ratio,
          extent and solidity to reject shapes that are unlikely to be
          potholes (long thin cracks along lane-line direction, small
          specks, road-sign shadows, etc.).
        - Each surviving contour is scored 0-1 by how strongly its shape
          statistics match "typical pothole" ranges and reported as a
          Detection with that confidence.

2. **Pluggable ML classifier hook (optional, for future real data).**
   `PotholeClassifier` extracts a small feature vector per candidate patch
   (area ratio, aspect ratio, extent, solidity, mean/std intensity,
   Laplacian variance as a texture-roughness proxy) and, if a trained
   scikit-learn model file is available (see `train_classifier.py` in this
   package for a template), uses it to re-score/filter the heuristic
   candidates instead of the fixed shape-based scoring rule. Without a
   trained model file present, it transparently falls back to the heuristic
   scorer above -- so `main.py` always runs, with or without a model.

This mirrors how a lot of real "pothole detection" student/prototype
projects are structured: classical CV for candidate generation (since
positive/negative road-damage patches are cheap to reason about
geometrically), with a slot left open for a learned classifier once labeled
data is collected.
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import numpy as np
import cv2


@dataclass
class Detection:
    """A single detected damage region, in pixel coordinates of the frame
    it was found in."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str = "pothole"

    @property
    def box(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def _road_roi_mask(shape, top_frac: float = 0.45, bottom_frac: float = 0.98
                    ) -> np.ndarray:
    """Wide trapezoid covering the drivable road surface (wider than the
    lane-only ROI used in lane_detection.py, since damage can appear across
    the whole lane width, not just near the marked lines)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    top_y = int(h * top_frac)
    bottom_y = int(h * bottom_frac)
    pts = np.array([
        [int(w * 0.02), bottom_y],
        [int(w * 0.30), top_y],
        [int(w * 0.70), top_y],
        [int(w * 0.98), bottom_y],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _shape_features(contour, frame_area: float, gray: np.ndarray):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = max(w * h, 1)
    extent = area / rect_area
    aspect_ratio = w / float(h) if h else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    area_ratio = area / frame_area

    patch = gray[y:y + h, x:x + w]
    mean_intensity = float(patch.mean()) if patch.size else 0.0
    std_intensity = float(patch.std()) if patch.size else 0.0
    laplacian_var = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size else 0.0

    return {
        "area": area,
        "x": x, "y": y, "w": w, "h": h,
        "extent": extent,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "area_ratio": area_ratio,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "laplacian_var": laplacian_var,
    }


def _heuristic_score(feat: dict) -> float:
    """Rule-based confidence in [0, 1] that a candidate blob is a pothole,
    based on shape statistics typical of pothole imagery: roughly blob-like
    (not a thin crack), moderate size relative to the frame, irregular
    boundary (lower solidity than a clean circle/ellipse), and rough
    internal texture (higher local variance from broken asphalt/shadow)."""
    score = 0.0

    # Size: too small = noise/gravel, too large = probably not a pothole
    if 0.0008 < feat["area_ratio"] < 0.05:
        score += 0.30
    elif 0.0003 < feat["area_ratio"] <= 0.0008:
        score += 0.10

    # Roughly blob-shaped, not an elongated crack/lane-marking sliver
    if 0.4 <= feat["aspect_ratio"] <= 2.5:
        score += 0.20

    # Irregular boundary (potholes rarely have a perfectly convex outline)
    if 0.55 <= feat["solidity"] <= 0.95:
        score += 0.20
    elif feat["solidity"] > 0.95:
        score += 0.05  # too clean/regular -- less likely, but not impossible

    # Fills a reasonable fraction of its bounding box (not a sparse cluster)
    if feat["extent"] > 0.35:
        score += 0.10

    # Textured/rough interior (cracked asphalt, shadow gradients)
    if feat["laplacian_var"] > 40:
        score += 0.20

    return float(min(score, 1.0))


class PotholeClassifier:
    """Optional learned re-scorer for candidate patches.

    Looks for a scikit-learn model pickled with joblib at `model_path`. If
    it isn't found (the default for this repo, since no labeled pothole
    dataset is bundled), `score()` transparently falls back to the
    heuristic rule above so the pipeline always works out of the box.

    See train_classifier.py for a minimal template showing how you would
    train and save such a model once you have labeled patch data (e.g. from
    the RDD2022 or a Kaggle pothole dataset).
    """

    def __init__(self, model_path: str = "models/pothole_classifier.joblib"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            try:
                import joblib  # optional dependency, only needed if a model exists
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None  # corrupt/incompatible model file -> fall back

    def _feature_vector(self, feat: dict) -> np.ndarray:
        return np.array([[
            feat["area_ratio"], feat["aspect_ratio"], feat["extent"],
            feat["solidity"], feat["mean_intensity"], feat["std_intensity"],
            feat["laplacian_var"],
        ]])

    def score(self, feat: dict) -> float:
        if self.model is not None:
            try:
                proba = self.model.predict_proba(self._feature_vector(feat))[0][1]
                return float(proba)
            except Exception:
                pass  # fall through to heuristic on any inference error
        return _heuristic_score(feat)


class PotholeDetector:
    """End-to-end candidate-generation + scoring pipeline for one frame."""

    def __init__(self,
                 min_confidence: float = 0.45,
                 block_size: int = 25,
                 c: int = 7,
                 model_path: str = "models/pothole_classifier.joblib"):
        self.min_confidence = min_confidence
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c
        self.classifier = PotholeClassifier(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        roi_mask = _road_roi_mask(frame.shape)

        # Adaptive threshold picks out locally-darker regions (shadows /
        # cracked asphalt inside a pothole) regardless of overall lighting.
        # Important: threshold the FULL blurred frame first, then apply the
        # ROI mask afterward. Masking before thresholding would zero out
        # everything outside the ROI, and the artificial road/black boundary
        # that creates gets picked up by the local adaptive threshold as a
        # spurious "dark region" running along the entire ROI edge.
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.c,
        )
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)
        detections: List[Detection] = []
        for c in contours:
            feat = _shape_features(c, frame_area, gray)
            if feat["area"] < 60:
                continue

            # Hard shape gates, applied before scoring. These reject shapes
            # that are geometrically not pothole-like regardless of how
            # they score on the softer heuristic below -- in particular,
            # long thin contours (lane-marking edges, dashed center-line
            # segments) can otherwise slip past the score because a large
            # *bounding box* along a diagonal, or a tall thin rectangle,
            # still lands in an acceptable size/extent range.
            if feat["w"] > 0.35 * frame_w or feat["h"] > 0.35 * frame_h:
                continue  # bounding box too large to be a (nearby) pothole
            if feat["extent"] < 0.15:
                continue  # almost no fill relative to its bbox -> thin line fragment
            if not (0.3 <= feat["aspect_ratio"] <= 3.2):
                continue  # too needle-like (tall/thin or wide/flat) to be a pothole blob

            confidence = self.classifier.score(feat)
            if confidence >= self.min_confidence:
                detections.append(Detection(
                    x=feat["x"], y=feat["y"], w=feat["w"], h=feat["h"],
                    confidence=confidence,
                ))

        detections = self._non_max_suppress(detections)
        return detections

    @staticmethod
    def _non_max_suppress(detections: List[Detection],
                           iou_threshold: float = 0.3) -> List[Detection]:
        """Simple greedy NMS so overlapping/duplicate contours around the
        same damage region don't produce multiple boxes."""
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []

        def iou(a: Detection, b: Detection) -> float:
            ax1, ay1, ax2, ay2 = a.box
            bx1, by1, bx2, by2 = b.box
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                return 0.0
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / float(area_a + area_b - inter)

        for det in detections:
            if all(iou(det, k) < iou_threshold for k in kept):
                kept.append(det)
        return kept

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"pothole {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 6, y1), (0, 0, 255), -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out
