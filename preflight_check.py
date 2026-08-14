from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

PATHS = {
    "YOLO model": Path(r"C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat\runs\detect\baby_person_empty_seat\weights\best.pt"),
    "LSTM model": Path(r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks\lstm_behavior_model.keras"),
    "LSTM scaler": Path(r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks\lstm_feature_scaler.pkl"),
    "LSTM label encoder": Path(r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks\lstm_label_encoder.pkl"),
    "LSTM feature columns": Path(r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks\lstm_feature_columns.json"),
    "AST model folder": Path(r"C:\Users\Admin\Desktop\AST_Audio_Project\ast_audio_model"),
    "XGBoost SHAP package": PROJECT_DIR / "best_tree_model_for_shap.pkl",
    "Dashboard HTML": PROJECT_DIR / "templates" / "index.html",
}


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")


def main() -> int:
    print("=" * 72)
    print("LIVE SHAP DASHBOARD PREFLIGHT CHECK")
    print("=" * 72)
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])

    errors: list[str] = []

    required_modules = [
        "cv2", "joblib", "keras", "mediapipe", "numpy", "pandas",
        "sounddevice", "tensorflow", "torch", "xgboost", "flask",
        "transformers", "ultralytics", "sklearn",
    ]

    print("\n1. Package imports")
    for module_name in required_modules:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "installed")
            ok(f"{module_name}: {version}")
        except Exception as exc:
            message = f"Cannot import {module_name}: {exc}"
            fail(message)
            errors.append(message)

    print("\n2. Required files and folders")
    for name, path in PATHS.items():
        if path.exists():
            ok(f"{name}: {path}")
        else:
            message = f"Missing {name}: {path}"
            fail(message)
            errors.append(message)

    if errors:
        print("\nPreflight stopped before model testing because required items are missing.")
        return 1

    print("\n3. LSTM metadata")
    try:
        columns = json.loads(PATHS["LSTM feature columns"].read_text(encoding="utf-8"))
        if not isinstance(columns, list) or len(columns) != 143:
            raise ValueError(f"Expected 143 LSTM features, found {len(columns) if isinstance(columns, list) else 'invalid JSON type'}")
        ok("LSTM feature list contains 143 columns")
    except Exception as exc:
        message = f"LSTM feature metadata error: {exc}"
        fail(message)
        errors.append(message)

    print("\n4. XGBoost prediction and native SHAP")
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        package = joblib.load(PATHS["XGBoost SHAP package"])
        required_keys = {
            "pipeline", "label_encoder", "feature_columns", "numeric_features",
            "categorical_features", "class_names", "temporal_window",
        }
        missing = required_keys.difference(package)
        if missing:
            raise KeyError(f"Missing model keys: {sorted(missing)}")

        pipeline = package["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        feature_columns = list(package["feature_columns"])

        row = {column: 0.0 for column in feature_columns}
        row["lstm_behavior"] = "Normal Movement"
        row["audio_class"] = "Normal"
        frame = pd.DataFrame([row], columns=feature_columns)

        prediction = pipeline.predict(frame)
        probabilities = pipeline.predict_proba(frame)
        processed = preprocessor.transform(frame)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
        processed = np.asarray(processed, dtype=np.float32)

        contributions = classifier.get_booster().predict(
            xgb.DMatrix(processed),
            pred_contribs=True,
            strict_shape=True,
        )
        contributions = np.asarray(contributions)

        if contributions.ndim != 3:
            raise ValueError(f"Unexpected native SHAP shape: {contributions.shape}")

        ok(f"Risk prediction works: encoded class {int(prediction[0])}")
        ok(f"Probability output shape: {probabilities.shape}")
        ok(f"Processed feature shape: {processed.shape}")
        ok(f"Native SHAP contribution shape: {contributions.shape}")
    except Exception as exc:
        message = f"XGBoost/native SHAP test failed: {exc}"
        fail(message)
        traceback.print_exc()
        errors.append(message)

    print("\n" + "=" * 72)
    if errors:
        print(f"PREFLIGHT FAILED: {len(errors)} problem(s) found.")
        print("Do not run app.py until these errors are fixed.")
        return 1

    print("PREFLIGHT PASSED")
    print("Run app.py with the same Python environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
