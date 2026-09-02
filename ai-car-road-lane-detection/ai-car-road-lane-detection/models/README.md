# models/

Empty by default. `PotholeDetector` runs entirely on the CV heuristic in
`src/pothole_detection.py` without needing anything here.

If you train the optional classifier (see `src/train_classifier.py`), the
saved model goes here as `pothole_classifier.joblib` and will be picked up
automatically on the next run.
