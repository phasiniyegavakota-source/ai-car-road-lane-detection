"""
train_classifier.py
--------------------
Template for training an optional ML classifier to replace the fixed-rule
heuristic scorer in pothole_detection.PotholeClassifier, once you have a
labeled dataset of road-surface patches (e.g. a public pothole dataset, or
patches you crop and label yourself from real dashcam footage).

This is intentionally NOT run as part of the default pipeline -- it is a
starting point to extend the project with a trained-model component, which
is why pothole_detection.py works fine without ever running this script.

Expected input
==============
A CSV with one row per labeled candidate patch and the columns:
    area_ratio, aspect_ratio, extent, solidity, mean_intensity,
    std_intensity, laplacian_var, label   (label: 1 = pothole, 0 = not)

You can generate these feature rows by running PotholeDetector's internal
`_shape_features` helper (or your own patch-feature extraction) over
labeled crops, and writing them to a CSV as you go.

Usage
=====
    python src/train_classifier.py --data path/to/labeled_features.csv \
        --out models/pothole_classifier.joblib

Requires scikit-learn and joblib (`pip install scikit-learn joblib`), which
are optional extras -- not required to run the base pipeline.
"""

import argparse
import os


FEATURE_COLUMNS = [
    "area_ratio", "aspect_ratio", "extent", "solidity",
    "mean_intensity", "std_intensity", "laplacian_var",
]


def main():
    parser = argparse.ArgumentParser(description="Train the optional pothole patch classifier.")
    parser.add_argument("--data", required=True, help="CSV of labeled patch features (see module docstring).")
    parser.add_argument("--out", default="models/pothole_classifier.joblib",
                         help="Where to save the trained model.")
    args = parser.parse_args()

    try:
        import pandas as pd
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except ImportError as e:
        raise SystemExit(
            "This training script needs pandas, scikit-learn, and joblib.\n"
            "Install with: pip install pandas scikit-learn joblib\n"
            f"(missing: {e.name})"
        )

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved trained model to: {args.out}")


if __name__ == "__main__":
    main()
