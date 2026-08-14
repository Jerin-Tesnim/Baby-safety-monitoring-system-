import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
import math
import shutil
import tempfile
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

import cv2
import joblib
import keras
import mediapipe as mp
import numpy as np
import pandas as pd
import shap
import sounddevice as sd
import tensorflow as tf
import torch
import xgboost as xgb

from flask import Flask, Response, jsonify, render_template
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from ultralytics import YOLO


# ============================================================
# 1. FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ============================================================
# 2. EXACT MODEL PATHS FROM THE FINAL PROJECT
# ============================================================

YOLO_MODEL_PATH = Path(
    r"C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat"
    r"\runs\detect\baby_person_empty_seat"
    r"\weights\best.pt"
)

LSTM_MODEL_PATH = Path(
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_behavior_model.keras"
)

LSTM_SCALER_PATH = Path(
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_feature_scaler.pkl"
)

LSTM_LABEL_ENCODER_PATH = Path(
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_label_encoder.pkl"
)

LSTM_FEATURE_COLUMNS_PATH = Path(
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_feature_columns.json"
)

AST_MODEL_FOLDER = Path(
    r"C:\Users\Admin\Desktop\AST_Audio_Project"
    r"\ast_audio_model"
)

# Put the uploaded model file in this project folder.
RISK_MODEL_PATH = Path(__file__).resolve().parent / "best_tree_model_for_shap.pkl"


# ============================================================
# 3. CAMERA, AUDIO AND MODEL SETTINGS
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
# Run YOLO with a low internal threshold, then apply a separate threshold
# per class. Empty-seat images are often harder than baby images.
YOLO_INFERENCE_CONFIDENCE = 0.40
BABY_CONFIDENCE_THRESHOLD = 0.40
EMPTY_SEAT_CONFIDENCE_THRESHOLD = 0.40
YOLO_RESULT_HOLD_SECONDS = 0.0

LSTM_SEQUENCE_LENGTH = 30
LSTM_ROLLING_WINDOW = 30

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3.0
AUDIO_CHANNELS = 1
MICROPHONE_DEVICE_INDEX = 1
# Audio below this RMS level is treated as silence/normal instead of noise.
AUDIO_SILENCE_RMS_THRESHOLD = 0.009

# Risk model was inspected from best_tree_model_for_shap.pkl.
RISK_TEMPORAL_WINDOW = 15
RISK_UPDATE_INTERVAL_SECONDS = 0.50
SHAP_UPDATE_INTERVAL_SECONDS = 0.50
MAX_SHAP_FEATURES = 8

# Save one CSV row only when a new risk/SHAP result is produced.
LIVE_CSV_PATH = Path(__file__).resolve().parent / "live_multimodal_risk_output.csv"
CSV_SHAP_FEATURE_COUNT = 3

EPSILON = 1e-8


# ============================================================
# 4. CLASS NAMES
# ============================================================

BABY_CLASS_NAMES = {
    "baby",
    "baby detection",
    "baby-detection",
    "baby_detection",
    "infant",
}

EMPTY_SEAT_CLASS_NAMES = {
    "empty seat",
    "empty-seat",
    "empty_seat",
}

EXPECTED_BEHAVIOR_CLASSES = [
    "Inactive",
    "Low Movement",
    "Normal Movement",
    "Restless Movement",
    "Sudden Movement",
]

EXPECTED_AUDIO_CLASSES = [
    "Baby_Cry",
    "Noise",
    "Normal",
]


# ============================================================
# 5. RAW LIVE FEATURES
# ============================================================

RAW_NUMERIC_FEATURES = [
    "baby_detected",
    "empty_seat_detected",
    "baby_count",
    "empty_seat_count",
    "yolo_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "pose_detected",
    "movement_score",
    "behavior_confidence",
    "audio_confidence",
    "cry_probability",
    "noise_probability",
    "normal_probability",
]

ROLLING_SOURCE_FEATURES = [
    "baby_detected",
    "empty_seat_detected",
    "baby_count",
    "yolo_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "pose_detected",
    "movement_score",
    "behavior_confidence",
    "audio_confidence",
    "cry_probability",
    "noise_probability",
    "normal_probability",
]


# ============================================================
# 6. SHARED THREAD-SAFE LIVE STATE
# ============================================================

state_lock = threading.Lock()
audio_lock = threading.Lock()
frame_lock = threading.Lock()

stop_event = threading.Event()

latest_audio_result = {
    "audio_class": "Normal",
    "audio_confidence": 0.0,
    "cry_probability": 0.0,
    "noise_probability": 0.0,
    "normal_probability": 1.0,
}

latest_live_state = {
    "system_ready": False,
    "message": "Loading models",
    "sample_id": 0,

    "detected_class": "None",
    "baby_detected": 0,
    "empty_seat_detected": 0,
    "baby_count": 0,
    "empty_seat_count": 0,
    "yolo_confidence": 0.0,
    "bbox_width": 0.0,
    "bbox_height": 0.0,
    "bbox_area": 0.0,

    "pose_detected": 0,
    "movement_score": 0.0,
    "lstm_behavior": "Collecting Sequence",
    "behavior_confidence": 0.0,

    "audio_class": "Normal",
    "audio_confidence": 0.0,
    "cry_probability": 0.0,
    "noise_probability": 0.0,
    "normal_probability": 1.0,

    "risk_class": "Waiting",
    "risk_confidence": 0.0,
    "risk_probabilities": {},
    "risk_explanation": "Waiting for enough live samples.",
    "shap_features": [],
    "fusion_status": "Waiting",
    "fusion_formula": "Waiting for live signals",

    "camera_ok": False,
    "audio_ok": False,
    "error": "",
}

latest_encoded_frame = None

# Keep the most recent valid YOLO result briefly to prevent the dashboard
# from flickering to "No Detection" when one frame is missed.
last_valid_yolo_result = None
last_valid_yolo_time = 0.0

# Last 15 synchronized live observations are kept only in RAM.
risk_history = deque(maxlen=RISK_TEMPORAL_WINDOW)

# LSTM state.
lstm_sequence_buffer = deque(maxlen=LSTM_SEQUENCE_LENGTH)
movement_history = deque(maxlen=LSTM_ROLLING_WINDOW)


# ============================================================
# 7. MODEL OBJECTS
# ============================================================

yolo_model = None
pose_model = None

lstm_model = None
lstm_scaler = None
lstm_label_encoder = None
lstm_feature_columns = None

ast_feature_extractor = None
ast_model = None
ast_device = None

risk_model_package = None
risk_pipeline = None
risk_preprocessor = None
risk_classifier = None
risk_label_encoder = None
risk_feature_columns = None
risk_numeric_features = None
risk_categorical_features = None
risk_class_names = None
risk_explainer = None
processed_feature_names = None


# ============================================================
# 8. BASIC HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return number
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def class_name_matches(class_name, allowed_names):
    normalized = normalize_label(class_name)
    allowed = {normalize_label(item) for item in allowed_names}
    return normalized in allowed


def check_required_path(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} was not found:\n{path}\n"
        )


def clean_behavior_label(label):
    normalized = normalize_label(label)

    mapping = {
        "inactive": "Inactive",
        "low movement": "Low Movement",
        "normal movement": "Normal Movement",
        "restless movement": "Restless Movement",
        "sudden movement": "Sudden Movement",
        "collecting sequence": "Collecting Sequence",
    }

    return mapping.get(normalized, str(label).strip())


def clean_audio_label(label):
    normalized = normalize_label(label)

    mapping = {
        "baby cry": "Baby_Cry",
        "babycry": "Baby_Cry",
        "cry": "Baby_Cry",
        "noise": "Noise",
        "normal": "Normal",
    }

    return mapping.get(normalized, str(label).strip())


def copy_live_state():
    with state_lock:
        copied = dict(latest_live_state)
        copied["risk_probabilities"] = dict(
            latest_live_state.get("risk_probabilities", {})
        )
        copied["shap_features"] = list(
            latest_live_state.get("shap_features", [])
        )
        return copied


def update_live_state(**values):
    with state_lock:
        latest_live_state.update(values)


# ============================================================
# 9. LOAD LSTM FEATURE COLUMN NAMES
# ============================================================

def load_lstm_feature_columns():
    with open(
        LSTM_FEATURE_COLUMNS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        return list(data)

    if isinstance(data, dict):
        for key in [
            "feature_columns",
            "features",
            "columns",
            "input_features",
        ]:
            if key in data:
                return list(data[key])

    raise ValueError(
        "Could not read LSTM feature columns from "
        f"{LSTM_FEATURE_COLUMNS_PATH}"
    )


# ============================================================
# 10. KERAS 3 TO KERAS 2.15 COMPATIBLE LSTM LOADER
# ============================================================

def _read_h5_dataset(h5_file, path):
    if path not in h5_file:
        raise KeyError(
            f"LSTM weight was not found inside model archive: {path}"
        )

    return np.asarray(h5_file[path])


def _build_tf215_lstm_model():
    model = keras.Sequential(
        [
            keras.layers.Input(
                shape=(30, 143),
                name="input_layer",
            ),
            keras.layers.LSTM(
                128,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                name="lstm",
            ),
            keras.layers.BatchNormalization(
                name="batch_normalization"
            ),
            keras.layers.Dropout(
                0.30,
                name="dropout",
            ),
            keras.layers.LSTM(
                64,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                name="lstm_1",
            ),
            keras.layers.BatchNormalization(
                name="batch_normalization_1"
            ),
            keras.layers.Dropout(
                0.30,
                name="dropout_1",
            ),
            keras.layers.LSTM(
                32,
                return_sequences=False,
                activation="tanh",
                recurrent_activation="sigmoid",
                name="lstm_2",
            ),
            keras.layers.Dropout(
                0.25,
                name="dropout_2",
            ),
            keras.layers.Dense(
                64,
                activation="relu",
                name="dense",
            ),
            keras.layers.Dropout(
                0.25,
                name="dropout_3",
            ),
            keras.layers.Dense(
                5,
                activation="softmax",
                name="dense_1",
            ),
        ],
        name="sequential",
    )

    model(
        np.zeros((1, 30, 143), dtype=np.float32),
        training=False,
    )

    return model


def load_compatible_lstm_model(model_path):
    model_path = Path(model_path)

    try:
        model = keras.models.load_model(
            str(model_path),
            compile=False,
        )
        print("LSTM model loaded directly.")
        return model

    except Exception as direct_error:
        print(
            "Direct LSTM loading failed. "
            "Trying Keras 3 to TensorFlow 2.15 conversion."
        )
        print("Direct loading reason:", direct_error)

    if not zipfile.is_zipfile(model_path):
        raise RuntimeError(
            "The LSTM model is not a valid .keras archive:\n"
            f"{model_path}"
        )

    import h5py

    temporary_folder = Path(
        tempfile.mkdtemp(prefix="lstm_live_shap_")
    )

    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            archive.extract(
                "model.weights.h5",
                temporary_folder,
            )

        weights_path = temporary_folder / "model.weights.h5"
        model = _build_tf215_lstm_model()

        with h5py.File(weights_path, "r") as h5_file:
            model.get_layer("lstm").set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm/cell/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm/cell/vars/1",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm/cell/vars/2",
                    ),
                ]
            )

            model.get_layer(
                "batch_normalization"
            ).set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization/vars/1",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization/vars/2",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization/vars/3",
                    ),
                ]
            )

            model.get_layer("lstm_1").set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm_1/cell/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm_1/cell/vars/1",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm_1/cell/vars/2",
                    ),
                ]
            )

            model.get_layer(
                "batch_normalization_1"
            ).set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization_1/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization_1/vars/1",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization_1/vars/2",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/batch_normalization_1/vars/3",
                    ),
                ]
            )

            model.get_layer("lstm_2").set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm_2/cell/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm_2/cell/vars/1",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/lstm_2/cell/vars/2",
                    ),
                ]
            )

            model.get_layer("dense").set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/dense/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/dense/vars/1",
                    ),
                ]
            )

            model.get_layer("dense_1").set_weights(
                [
                    _read_h5_dataset(
                        h5_file,
                        "layers/dense_1/vars/0",
                    ),
                    _read_h5_dataset(
                        h5_file,
                        "layers/dense_1/vars/1",
                    ),
                ]
            )

        print("Keras 3 LSTM weights loaded successfully.")
        return model

    finally:
        shutil.rmtree(
            temporary_folder,
            ignore_errors=True,
        )


# ============================================================
# 11. RISK MODEL FEATURE-NAME HELPERS
# ============================================================

def get_processed_feature_names(preprocessor, fallback_count):
    try:
        names = list(
            preprocessor.get_feature_names_out()
        )

        cleaned = []

        for name in names:
            text = str(name)

            if "__" in text:
                text = text.split("__", 1)[1]

            cleaned.append(text)

        return cleaned

    except Exception:
        return [
            f"processed_feature_{index}"
            for index in range(fallback_count)
        ]


def modality_for_feature(feature_name):
    lower = str(feature_name).lower()

    if any(token in lower for token in [
        "cry",
        "noise",
        "normal_probability",
        "audio",
    ]):
        return "AST audio"

    if any(token in lower for token in [
        "behavior",
        "inactive",
        "restless",
        "sudden",
        "low_movement",
        "normal_movement",
        "lstm",
    ]):
        return "LSTM behavior"

    if any(token in lower for token in [
        "movement",
        "pose",
    ]):
        return "MediaPipe movement"

    if any(token in lower for token in [
        "baby",
        "empty_seat",
        "yolo",
        "bbox",
    ]):
        return "YOLO vision"

    return "Other"


def readable_feature_name(feature_name):
    text = str(feature_name)

    for prefix in [
        "numeric__",
        "categorical__",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    replacements = {
        "_roll_mean": " recent average",
        "_roll_std": " recent variation",
        "_roll_min": " recent minimum",
        "_roll_max": " recent maximum",
        "_change": " latest change",
        "_recent_ratio": " recent ratio",
    }

    for suffix, replacement in replacements.items():
        if text.endswith(suffix):
            text = text[:-len(suffix)] + replacement
            break

    text = text.replace("_", " ")
    text = " ".join(text.split())

    if not text:
        return str(feature_name)

    return text[0].upper() + text[1:]


# ============================================================
# 12. LOAD EXACT XGBOOST + PREPROCESSOR + SHAP PACKAGE
# ============================================================

def load_risk_model_package():
    global risk_model_package
    global risk_pipeline
    global risk_preprocessor
    global risk_classifier
    global risk_label_encoder
    global risk_feature_columns
    global risk_numeric_features
    global risk_categorical_features
    global risk_class_names
    global risk_explainer
    global processed_feature_names
    global RISK_TEMPORAL_WINDOW
    global risk_history

    check_required_path(
        RISK_MODEL_PATH,
        "XGBoost SHAP model package",
    )

    risk_model_package = joblib.load(
        RISK_MODEL_PATH
    )

    required_keys = {
        "pipeline",
        "label_encoder",
        "feature_columns",
        "numeric_features",
        "categorical_features",
        "class_names",
        "temporal_window",
    }

    missing_keys = required_keys.difference(
        risk_model_package.keys()
    )

    if missing_keys:
        raise KeyError(
            "Missing keys inside best_tree_model_for_shap.pkl: "
            + ", ".join(sorted(missing_keys))
        )

    risk_pipeline = risk_model_package["pipeline"]
    risk_label_encoder = risk_model_package["label_encoder"]
    risk_feature_columns = list(
        risk_model_package["feature_columns"]
    )
    risk_numeric_features = list(
        risk_model_package["numeric_features"]
    )
    risk_categorical_features = list(
        risk_model_package["categorical_features"]
    )
    risk_class_names = list(
        risk_model_package["class_names"]
    )

    RISK_TEMPORAL_WINDOW = int(
        risk_model_package["temporal_window"]
    )

    risk_history = deque(
        maxlen=RISK_TEMPORAL_WINDOW
    )

    if "preprocessor" not in risk_pipeline.named_steps:
        raise KeyError(
            "The risk pipeline does not contain a "
            "'preprocessor' step."
        )

    if "classifier" not in risk_pipeline.named_steps:
        raise KeyError(
            "The risk pipeline does not contain a "
            "'classifier' step."
        )

    risk_preprocessor = (
        risk_pipeline.named_steps["preprocessor"]
    )

    risk_classifier = (
        risk_pipeline.named_steps["classifier"]
    )

    if not hasattr(risk_classifier, "get_booster"):
        raise TypeError(
            "The classifier inside the model package "
            "is not an XGBoost classifier."
        )

    # Create one valid dummy row to discover the exact processed feature count.
    dummy_row = {
        column: 0.0
        for column in risk_feature_columns
    }

    dummy_row["lstm_behavior"] = "Normal Movement"
    dummy_row["audio_class"] = "Normal"

    dummy_frame = pd.DataFrame(
        [dummy_row],
        columns=risk_feature_columns,
    )

    processed_dummy = risk_preprocessor.transform(
        dummy_frame
    )

    if hasattr(processed_dummy, "toarray"):
        processed_dummy = processed_dummy.toarray()

    processed_dummy = np.asarray(
        processed_dummy,
        dtype=np.float32,
    )

    processed_feature_names = (
        get_processed_feature_names(
            risk_preprocessor,
            processed_dummy.shape[1],
        )
    )

    # SHAP TreeExplainer 0.46 cannot parse the vector-valued multiclass
    # base score used by XGBoost 3.2. Native XGBoost contributions are
    # used later with pred_contribs=True instead.
    risk_explainer = None

    print("\nRisk model loaded successfully.")
    print("Model name:", risk_model_package["model_name"])
    print("Raw feature count:", len(risk_feature_columns))
    print(
        "Processed XGBoost feature count:",
        processed_dummy.shape[1],
    )
    print("Risk classes:", risk_class_names)
    print("Temporal window:", RISK_TEMPORAL_WINDOW)
    print(
        "Saved test accuracy:",
        f"{risk_model_package.get('test_accuracy', 0.0):.4f}",
    )


# ============================================================
# 13. TEMPORAL FEATURE CREATION IN RAM
# ============================================================

def create_raw_risk_row(
    yolo_result,
    pose_detected,
    movement_score,
    behavior,
    behavior_confidence,
    audio_result,
):
    return {
        "baby_detected": safe_int(
            yolo_result.get("baby_detected", 0)
        ),
        "empty_seat_detected": safe_int(
            yolo_result.get("empty_seat_detected", 0)
        ),
        "baby_count": safe_int(
            yolo_result.get("baby_count", 0)
        ),
        "empty_seat_count": safe_int(
            yolo_result.get("empty_seat_count", 0)
        ),
        "yolo_confidence": safe_float(
            yolo_result.get("yolo_confidence", 0.0)
        ),
        "bbox_width": safe_float(
            yolo_result.get("bbox_width", 0.0)
        ),
        "bbox_height": safe_float(
            yolo_result.get("bbox_height", 0.0)
        ),
        "bbox_area": safe_float(
            yolo_result.get("bbox_area", 0.0)
        ),
        "pose_detected": safe_int(
            pose_detected
        ),
        "movement_score": safe_float(
            movement_score
        ),
        "behavior_confidence": safe_float(
            behavior_confidence
        ),
        "audio_confidence": safe_float(
            audio_result.get("audio_confidence", 0.0)
        ),
        "cry_probability": safe_float(
            audio_result.get("cry_probability", 0.0)
        ),
        "noise_probability": safe_float(
            audio_result.get("noise_probability", 0.0)
        ),
        "normal_probability": safe_float(
            audio_result.get("normal_probability", 0.0)
        ),
        "lstm_behavior": clean_behavior_label(
            behavior
        ),
        "audio_class": clean_audio_label(
            audio_result.get("audio_class", "Normal")
        ),
    }


def append_live_risk_row(raw_row):
    # No CSV is written here. The row exists only in memory.
    risk_history.append(dict(raw_row))


def _series_statistics(values):
    numeric = pd.Series(
        values,
        dtype="float64",
    )

    latest = safe_float(numeric.iloc[-1])

    if len(numeric) > 1:
        standard_deviation = safe_float(
            numeric.std(ddof=1)
        )
        change = safe_float(
            numeric.iloc[-1] - numeric.iloc[-2]
        )
    else:
        standard_deviation = 0.0
        change = 0.0

    return {
        "mean": safe_float(numeric.mean()),
        "std": standard_deviation,
        "min": safe_float(numeric.min()),
        "max": safe_float(numeric.max()),
        "change": change,
        "latest": latest,
    }


def _recent_ratio(column_name, target_label):
    if not risk_history:
        return 0.0

    matches = 0

    for row in risk_history:
        value = normalize_label(
            row.get(column_name, "")
        )

        if value == normalize_label(target_label):
            matches += 1

    return matches / len(risk_history)


def build_current_risk_feature_frame():
    if not risk_history:
        raise RuntimeError(
            "No live observations are available for risk prediction."
        )

    latest_row = dict(risk_history[-1])
    feature_row = {}

    # Current numeric values.
    for column in RAW_NUMERIC_FEATURES:
        feature_row[column] = safe_float(
            latest_row.get(column, 0.0)
        )

    # Rolling and change values expected by the trained model.
    for column in ROLLING_SOURCE_FEATURES:
        values = [
            safe_float(row.get(column, 0.0))
            for row in risk_history
        ]

        statistics = _series_statistics(values)

        feature_row[f"{column}_roll_mean"] = (
            statistics["mean"]
        )
        feature_row[f"{column}_roll_std"] = (
            statistics["std"]
        )
        feature_row[f"{column}_roll_min"] = (
            statistics["min"]
        )
        feature_row[f"{column}_roll_max"] = (
            statistics["max"]
        )
        feature_row[f"{column}_change"] = (
            statistics["change"]
        )

    # Recent AST class ratios.
    feature_row["audio_cry_recent_ratio"] = (
        _recent_ratio("audio_class", "Baby_Cry")
    )
    feature_row["audio_noise_recent_ratio"] = (
        _recent_ratio("audio_class", "Noise")
    )
    feature_row["audio_normal_recent_ratio"] = (
        _recent_ratio("audio_class", "Normal")
    )

    # Recent LSTM class ratios.
    feature_row["behavior_inactive_recent_ratio"] = (
        _recent_ratio("lstm_behavior", "Inactive")
    )
    feature_row["behavior_low_movement_recent_ratio"] = (
        _recent_ratio("lstm_behavior", "Low Movement")
    )
    feature_row["behavior_normal_movement_recent_ratio"] = (
        _recent_ratio("lstm_behavior", "Normal Movement")
    )
    feature_row["behavior_restless_movement_recent_ratio"] = (
        _recent_ratio("lstm_behavior", "Restless Movement")
    )
    feature_row["behavior_sudden_movement_recent_ratio"] = (
        _recent_ratio("lstm_behavior", "Sudden Movement")
    )

    feature_row["lstm_behavior"] = clean_behavior_label(
        latest_row.get(
            "lstm_behavior",
            "Normal Movement",
        )
    )

    feature_row["audio_class"] = clean_audio_label(
        latest_row.get(
            "audio_class",
            "Normal",
        )
    )

    # Enforce the exact 96-column raw input order saved in the model package.
    ordered_row = {}

    for column in risk_feature_columns:
        if column in risk_categorical_features:
            if column == "lstm_behavior":
                ordered_row[column] = feature_row.get(
                    column,
                    "Normal Movement",
                )
            elif column == "audio_class":
                ordered_row[column] = feature_row.get(
                    column,
                    "Normal",
                )
            else:
                ordered_row[column] = str(
                    feature_row.get(column, "Unknown")
                )
        else:
            ordered_row[column] = safe_float(
                feature_row.get(column, 0.0)
            )

    return pd.DataFrame(
        [ordered_row],
        columns=risk_feature_columns,
    )


print("Part 1 definitions loaded.")
INACTIVE_THRESHOLD = 0.0025
LOW_MOVEMENT_THRESHOLD = 0.0060
RESTLESS_MEAN_THRESHOLD = 0.0120
RESTLESS_STD_THRESHOLD = 0.0040
SUDDEN_MOVEMENT_THRESHOLD = 0.0350
BEHAVIOR_WINDOW = 20


def load_all_live_models():
    global yolo_model, pose_model
    global lstm_model, lstm_scaler, lstm_label_encoder, lstm_feature_columns
    global LSTM_SEQUENCE_LENGTH, lstm_sequence_buffer
    global ast_feature_extractor, ast_model, ast_device

    required_items = [
        (YOLO_MODEL_PATH, "YOLO model"),
        (LSTM_MODEL_PATH, "LSTM model"),
        (LSTM_SCALER_PATH, "LSTM scaler"),
        (LSTM_LABEL_ENCODER_PATH, "LSTM label encoder"),
        (LSTM_FEATURE_COLUMNS_PATH, "LSTM feature columns"),
        (AST_MODEL_FOLDER, "AST model folder"),
        (RISK_MODEL_PATH, "XGBoost SHAP model package"),
    ]

    for path, name in required_items:
        check_required_path(path, name)

    print("\nLoading YOLO model.")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    print("YOLO classes:", yolo_model.names)

    print("\nLoading LSTM model and preprocessing files.")
    lstm_model = load_compatible_lstm_model(LSTM_MODEL_PATH)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    lstm_label_encoder = joblib.load(LSTM_LABEL_ENCODER_PATH)
    lstm_feature_columns = load_lstm_feature_columns()

    model_input_shape = lstm_model.input_shape
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    model_sequence_length = int(model_input_shape[1])
    model_feature_count = int(model_input_shape[2])

    if model_feature_count != len(lstm_feature_columns):
        raise ValueError(
            "\nLSTM feature mismatch.\n"
            f"Model expects: {model_feature_count}\n"
            f"Feature JSON contains: {len(lstm_feature_columns)}"
        )

    LSTM_SEQUENCE_LENGTH = model_sequence_length
    lstm_sequence_buffer = deque(maxlen=LSTM_SEQUENCE_LENGTH)

    print("LSTM input shape:", model_input_shape)
    print("LSTM classes:", list(lstm_label_encoder.classes_))

    print("\nLoading MediaPipe Pose.")
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    print("\nLoading AST audio model.")
    ast_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ast_feature_extractor = AutoFeatureExtractor.from_pretrained(str(AST_MODEL_FOLDER))
    ast_model = ASTForAudioClassification.from_pretrained(str(AST_MODEL_FOLDER))
    ast_model.to(ast_device)
    ast_model.eval()

    print("AST device:", ast_device)
    print("AST classes:", ast_model.config.id2label)

    print("\nLoading XGBoost and SHAP.")
    load_risk_model_package()

    update_live_state(
        message="Models loaded. Starting camera and microphone.",
        error="",
    )


def process_yolo(frame):
    predictions = yolo_model.predict(
        source=frame,
        conf=YOLO_INFERENCE_CONFIDENCE,
        imgsz=320,
        verbose=False,
    )

    result = predictions[0]
    display_frame = frame.copy()

    baby_boxes = []
    empty_seat_boxes = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = str(yolo_model.names[class_id])

            # Keep only the two YOLO classes used by the dashboard.
            # Person/other classes are ignored and therefore become "None".
            if (
                class_name_matches(class_name, BABY_CLASS_NAMES)
                and confidence >= BABY_CONFIDENCE_THRESHOLD
            ):
                baby_boxes.append((box, confidence))
            elif (
                class_name_matches(class_name, EMPTY_SEAT_CLASS_NAMES)
                and confidence >= EMPTY_SEAT_CONFIDENCE_THRESHOLD
            ):
                empty_seat_boxes.append((box, confidence))

    # Exact requested YOLO condition:
    # 1) Baby present      -> Baby
    # 2) No baby + chair empty -> Empty Seat
    # 3) Neither          -> None
    if baby_boxes:
        selected_boxes = baby_boxes
        selected_name = "Baby"
        box_color = (0, 220, 0)
    elif empty_seat_boxes:
        selected_boxes = empty_seat_boxes
        selected_name = "Empty Seat"
        box_color = (0, 165, 255)
    else:
        selected_boxes = []
        selected_name = "None"
        box_color = (0, 165, 255)

    selected_confidence = 0.0
    selected_bbox_width = 0.0
    selected_bbox_height = 0.0
    selected_bbox_area = 0.0

    for box, confidence in selected_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        bbox_width = max(0, int(x2 - x1))
        bbox_height = max(0, int(y2 - y1))
        bbox_area = bbox_width * bbox_height

        if confidence > selected_confidence:
            selected_confidence = confidence
            selected_bbox_width = bbox_width
            selected_bbox_height = bbox_height
            selected_bbox_area = bbox_area

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            display_frame,
            f"{selected_name} {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    baby_count = len(baby_boxes) if selected_name == "Baby" else 0
    empty_seat_count = len(empty_seat_boxes) if selected_name == "Empty Seat" else 0

    features = {
        "detected_class": selected_name,
        "baby_detected": int(selected_name == "Baby"),
        "empty_seat_detected": int(selected_name == "Empty Seat"),
        "baby_count": baby_count,
        "empty_seat_count": empty_seat_count,
        "yolo_confidence": round(selected_confidence, 6),
        "bbox_width": float(selected_bbox_width),
        "bbox_height": float(selected_bbox_height),
        "bbox_area": float(selected_bbox_area),
    }

    return display_frame, features


def extract_pose_row(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = pose_model.process(rgb)
    rgb.flags.writeable = True

    if result.pose_landmarks is None:
        return None, result

    row = {}
    for index, landmark in enumerate(result.pose_landmarks.landmark):
        row[f"landmark_{index}_x"] = float(landmark.x)
        row[f"landmark_{index}_y"] = float(landmark.y)
        row[f"landmark_{index}_z"] = float(landmark.z)
        row[f"landmark_{index}_visibility"] = float(landmark.visibility)

    row["pose_detected"] = 1
    return row, result


def draw_pose(frame, pose_result):
    if pose_result is not None and pose_result.pose_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
        )


def calculate_landmark_movement(previous_pose_row, current_pose_row):
    if previous_pose_row is None or current_pose_row is None:
        return 0.0

    stable_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    movements = []

    for index in stable_landmarks:
        previous_visibility = safe_float(previous_pose_row.get(f"landmark_{index}_visibility", 0.0))
        current_visibility = safe_float(current_pose_row.get(f"landmark_{index}_visibility", 0.0))

        if min(previous_visibility, current_visibility) < 0.50:
            continue

        previous_x = safe_float(previous_pose_row.get(f"landmark_{index}_x", 0.0))
        previous_y = safe_float(previous_pose_row.get(f"landmark_{index}_y", 0.0))
        current_x = safe_float(current_pose_row.get(f"landmark_{index}_x", 0.0))
        current_y = safe_float(current_pose_row.get(f"landmark_{index}_y", 0.0))

        distance = np.sqrt((current_x - previous_x) ** 2 + (current_y - previous_y) ** 2)
        movements.append(float(distance))

    return float(np.median(movements)) if movements else 0.0


def create_live_lstm_feature_row(pose_row):
    row = {}
    if pose_row is not None:
        row.update(pose_row)

    movement_values = np.asarray(list(movement_history), dtype=np.float32)
    if len(movement_values) == 0:
        movement_values = np.asarray([0.0], dtype=np.float32)

    movement_score = float(movement_values[-1])
    previous_movement = float(movement_values[-2]) if len(movement_values) >= 2 else movement_score
    movement_change = abs(movement_score - previous_movement)

    if len(movement_values) >= 3:
        previous_change = abs(float(movement_values[-2]) - float(movement_values[-3]))
    else:
        previous_change = 0.0

    movement_acceleration = abs(movement_change - previous_change)
    movement_velocity = movement_score - previous_movement
    rolling_values = movement_values[-LSTM_ROLLING_WINDOW:]
    rolling_mean = float(np.mean(rolling_values))
    rolling_std = float(np.std(rolling_values, ddof=1)) if len(rolling_values) > 1 else 0.0
    rolling_min = float(np.min(rolling_values))
    rolling_max = float(np.max(rolling_values))
    movement_range = rolling_max - rolling_min
    movement_ratio = movement_score / (rolling_mean + EPSILON)

    row.update({
        "movement_score": movement_score,
        "movement_change": movement_change,
        "movement_acceleration": movement_acceleration,
        "movement_velocity": movement_velocity,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_min": rolling_min,
        "rolling_max": rolling_max,
        "movement_range": movement_range,
        "movement_ratio": movement_ratio,
        "movement_rolling_mean": rolling_mean,
        "movement_rolling_std": rolling_std,
        "movement_rolling_min": rolling_min,
        "movement_rolling_max": rolling_max,
        "rolling_range": movement_range,
        "movement_mean_30": rolling_mean,
        "movement_std_30": rolling_std,
        "movement_min_30": rolling_min,
        "movement_max_30": rolling_max,
        "movement_range_30": movement_range,
    })

    return row


def prepare_lstm_input_array(live_feature_row):
    ordered_values = [safe_float(live_feature_row.get(column, 0.0)) for column in lstm_feature_columns]
    return np.asarray(ordered_values, dtype=np.float32).reshape(1, -1)


def predict_lstm_behavior():
    sequence_array = np.asarray(lstm_sequence_buffer, dtype=np.float32)
    sequence_array = np.expand_dims(sequence_array, axis=0)
    probabilities = np.asarray(lstm_model.predict(sequence_array, verbose=0))[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    if hasattr(lstm_label_encoder, "classes_"):
        behavior = str(lstm_label_encoder.classes_[class_index])
    else:
        behavior = EXPECTED_BEHAVIOR_CLASSES[class_index] if class_index < len(EXPECTED_BEHAVIOR_CLASSES) else f"Class_{class_index}"

    return clean_behavior_label(behavior), confidence


def correct_behavior_with_movement(lstm_behavior, lstm_confidence):
    recent_values = np.asarray(list(movement_history)[-BEHAVIOR_WINDOW:], dtype=np.float32)

    if len(recent_values) < 5:
        return clean_behavior_label(lstm_behavior), safe_float(lstm_confidence)

    average_movement = float(np.mean(recent_values))
    movement_std = float(np.std(recent_values))
    maximum_movement = float(np.max(recent_values))

    if maximum_movement >= SUDDEN_MOVEMENT_THRESHOLD:
        return "Sudden Movement", 0.95
    if average_movement >= RESTLESS_MEAN_THRESHOLD and movement_std >= RESTLESS_STD_THRESHOLD:
        return "Restless Movement", 0.90
    if average_movement < INACTIVE_THRESHOLD:
        return "Inactive", 0.95
    if average_movement < LOW_MOVEMENT_THRESHOLD:
        return "Low Movement", 0.90

    normalized_lstm = normalize_label(lstm_behavior)
    if normalized_lstm in {"inactive", "low movement", "normal movement", "restless movement", "sudden movement"}:
        return clean_behavior_label(lstm_behavior), safe_float(lstm_confidence)

    return "Normal Movement", max(safe_float(lstm_confidence), 0.70)


def record_audio():
    sample_count = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECONDS)
    audio = sd.rec(
        sample_count,
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="float32",
        device=MICROPHONE_DEVICE_INDEX,
    )
    sd.wait()
    return np.squeeze(audio).astype(np.float32)


def audio_rms(audio):
    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def get_probability_by_keyword(probability_map, keywords):
    for label, probability in probability_map.items():
        normalized_label = normalize_label(label)
        for keyword in keywords:
            if normalize_label(keyword) in normalized_label:
                return float(probability)
    return 0.0


def standardize_ast_audio_class(label):
    normalized = normalize_label(label)
    if "cry" in normalized:
        return "Baby_Cry"
    if "noise" in normalized:
        return "Noise"
    if "normal" in normalized:
        return "Normal"
    return str(label)


def predict_ast_audio(audio):
    # Very low-level/silent audio is treated as Normal.
    # Otherwise AST selects only among Baby_Cry, Noise and Normal.
    current_rms = audio_rms(audio)

    if current_rms < AUDIO_SILENCE_RMS_THRESHOLD:
        return {
            "audio_class": "Normal",
            "audio_confidence": 1.0,
            "cry_probability": 0.0,
            "noise_probability": 0.0,
            "normal_probability": 1.0,
        }

    inputs = ast_feature_extractor(
        audio,
        sampling_rate=AUDIO_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(ast_device)

    with torch.no_grad():
        output = ast_model(input_values=input_values)
        probabilities = torch.softmax(output.logits, dim=-1)[0].cpu().numpy()

    id2label = ast_model.config.id2label
    probability_map = {
        str(id2label.get(index, index)): float(probability)
        for index, probability in enumerate(probabilities)
    }

    cry_probability = get_probability_by_keyword(
        probability_map,
        ["cry", "baby cry"],
    )
    noise_probability = get_probability_by_keyword(
        probability_map,
        ["noise"],
    )
    normal_probability = get_probability_by_keyword(
        probability_map,
        ["normal"],
    )

    class_probabilities = {
        "Baby_Cry": cry_probability,
        "Noise": noise_probability,
        "Normal": normal_probability,
    }

    audio_class = max(class_probabilities, key=class_probabilities.get)
    confidence = float(class_probabilities[audio_class])

    return {
        "audio_class": audio_class,
        "audio_confidence": round(confidence, 6),
        "cry_probability": round(cry_probability, 6),
        "noise_probability": round(noise_probability, 6),
        "normal_probability": round(normal_probability, 6),
    }


def audio_processing_worker():
    global latest_audio_result

    while not stop_event.is_set():
        try:
            audio = record_audio()

            # AST output is restricted to the three trained classes:
            # Baby_Cry, Noise and Normal.
            result = predict_ast_audio(audio)

            with audio_lock:
                latest_audio_result = result.copy()
            update_live_state(audio_ok=True, error="")
        except Exception as error:
            error_text = f"AST microphone error: {error}"
            print(error_text)
            with audio_lock:
                latest_audio_result = {
                    "audio_class": "Normal",
                    "audio_confidence": 0.0,
                    "cry_probability": 0.0,
                    "noise_probability": 0.0,
                    "normal_probability": 1.0,
                }
            update_live_state(audio_ok=False, error=error_text)
            time.sleep(1.0)


def decode_risk_class(encoded_prediction):
    try:
        return str(risk_label_encoder.inverse_transform([int(encoded_prediction)])[0])
    except Exception:
        index = int(encoded_prediction)
        if 0 <= index < len(risk_class_names):
            return str(risk_class_names[index])
        return f"Class_{index}"


def clean_risk_label(label):
    normalized = normalize_label(label)
    mapping = {
        "high risk": "High Risk",
        "low risk": "Low Risk",
        "medium risk": "Medium Risk",
        "no baby": "No Baby",
    }
    return mapping.get(normalized, str(label).replace("_", " "))


def get_prediction_probabilities(input_frame):
    probabilities = risk_pipeline.predict_proba(input_frame)[0]
    probability_map = {}
    for class_index, probability in enumerate(probabilities):
        class_label = clean_risk_label(decode_risk_class(class_index))
        probability_map[class_label] = round(float(probability), 6)
    return probabilities, probability_map


def select_class_shap_values(shap_output, predicted_class_index, processed_feature_count):
    values = shap_output.values if hasattr(shap_output, "values") else shap_output

    if isinstance(values, list):
        selected = np.asarray(values[predicted_class_index])
        return (selected[0] if selected.ndim == 2 else selected).astype(np.float64)

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[1] == processed_feature_count:
            return values[0, :, predicted_class_index].astype(np.float64)
        if values.shape[0] == 1 and values.shape[2] == processed_feature_count:
            return values[0, predicted_class_index, :].astype(np.float64)
        if values.shape[0] == len(risk_class_names) and values.shape[2] == processed_feature_count:
            return values[predicted_class_index, 0, :].astype(np.float64)

    if values.ndim == 2:
        if values.shape[0] == 1:
            return values[0].astype(np.float64)
        if values.shape[0] == len(risk_class_names):
            return values[predicted_class_index].astype(np.float64)

    if values.ndim == 1:
        return values.astype(np.float64)

    raise ValueError(f"Unsupported SHAP output shape: {values.shape}")


def _current_output_labels(yolo_result, latest_row):
    detected = str(yolo_result.get("detected_class", "None")).replace("_", " ")
    if detected.lower() in {"none", "no detection", "unknown"}:
        detected = "No Detection"

    behavior = clean_behavior_label(
        latest_row.get("lstm_behavior", "Collecting Sequence")
    )
    audio = clean_audio_label(
        latest_row.get("audio_class", "Normal")
    )

    audio_display = {
        "Baby_Cry": "Baby Cry",
        "Noise": "Noise",
        "Normal": "Normal",
    }.get(audio, str(audio).replace("_", " "))

    return detected, behavior, audio_display


def determine_fusion_status(yolo_result, behavior, audio_display):
    """Return one of the 11 project-defined fusion conditions."""
    baby_detected = safe_int(yolo_result.get("baby_detected", 0)) == 1

    if not baby_detected:
        return "No Baby Detected", "No baby is currently detected."

    behavior_key = normalize_label(behavior)
    is_cry = normalize_label(audio_display) in {"baby cry", "baby_cry", "cry"}

    if is_cry:
        mapping = {
            "low movement": (
                "Possible Distress",
                "Baby + Low Movement + Baby Cry",
            ),
            "restless movement": (
                "Distressed Movement",
                "Baby + Restless Movement + Baby Cry",
            ),
            "sudden movement": (
                "High Distress",
                "Baby + Sudden Movement + Baby Cry",
            ),
            "inactive": (
                "Critical Inactivity",
                "Baby + Inactive + Baby Cry",
            ),
            "normal movement": (
                "Crying with Normal Movement",
                "Baby + Normal Movement + Baby Cry",
            ),
        }
    else:
        mapping = {
            "normal movement": (
                "Normal Condition",
                f"Baby + Normal Movement + {audio_display}",
            ),
            "low movement": (
                "Low Movement Condition",
                f"Baby + Low Movement + {audio_display}",
            ),
            "inactive": (
                "Inactive Condition",
                f"Baby + Inactive + {audio_display}",
            ),
            "restless movement": (
                "Restless Movement",
                f"Baby + Restless Movement + {audio_display}",
            ),
            "sudden movement": (
                "Sudden Movement",
                f"Baby + Sudden Movement + {audio_display}",
            ),
        }

    return mapping.get(
        behavior_key,
        (
            "Analysing Condition",
            f"Baby + {behavior} + {audio_display}",
        ),
    )




def risk_from_fusion_status(fusion_status):
    """Map the 11 project fusion conditions to the final dashboard risk level."""
    mapping = {
        "No Baby Detected": "No Baby",
        "Normal Condition": "Low Risk",
        "Low Movement Condition": "Medium Risk",
        "Inactive Condition": "High Risk",
        "Restless Movement": "Medium Risk",
        "Sudden Movement": "High Risk",
        "Possible Distress": "High Risk",
        "Distressed Movement": "High Risk",
        "High Distress": "High Risk",
        "Critical Inactivity": "High Risk",
        "Crying with Normal Movement": "Medium Risk",
    }
    return mapping.get(fusion_status, "Medium Risk")


def risk_class_index_for_label(target_label):
    target = normalize_label(target_label)
    for class_index in range(len(risk_class_names)):
        label = clean_risk_label(decode_risk_class(class_index))
        if normalize_label(label) == target:
            return class_index
    raise ValueError(f"Risk class not found in trained model: {target_label}")


def explain_current_prediction(input_frame, predicted_class_index, yolo_result):
    """Aggregate exact native XGBoost SHAP values into YOLO, Behaviour and AST."""
    processed = risk_preprocessor.transform(input_frame)

    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    processed = np.asarray(processed, dtype=np.float32)

    booster = risk_classifier.get_booster()
    dmatrix = xgb.DMatrix(processed)
    raw_contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        strict_shape=True,
    )

    contributions = np.asarray(raw_contributions)
    feature_count = processed.shape[1]
    class_count = len(risk_class_names)

    if (
        contributions.ndim == 3
        and contributions.shape[0] == 1
        and contributions.shape[1] == class_count
        and contributions.shape[2] == feature_count + 1
    ):
        selected_shap = contributions[0, predicted_class_index, :-1]
    elif (
        contributions.ndim == 3
        and contributions.shape[0] == 1
        and contributions.shape[1] == feature_count + 1
        and contributions.shape[2] == class_count
    ):
        selected_shap = contributions[0, :-1, predicted_class_index]
    elif (
        contributions.ndim == 2
        and contributions.shape[0] == 1
        and contributions.shape[1] == feature_count + 1
    ):
        selected_shap = contributions[0, :-1]
    else:
        raise ValueError(
            "Unsupported native XGBoost contribution shape: "
            f"{contributions.shape}"
        )

    selected_shap = np.asarray(selected_shap, dtype=np.float64)
    usable_count = min(len(selected_shap), len(processed_feature_names))

    grouped = {
        "YOLO": 0.0,
        "Behaviour": 0.0,
        "AST": 0.0,
    }

    for index in range(usable_count):
        value = safe_float(selected_shap[index])
        modality = modality_for_feature(processed_feature_names[index])

        if modality == "YOLO vision":
            grouped["YOLO"] += value
        elif modality in {"LSTM behavior", "MediaPipe movement"}:
            grouped["Behaviour"] += value
        elif modality == "AST audio":
            grouped["AST"] += value

    latest_row = risk_history[-1] if risk_history else {}
    detected, behavior, audio_display = _current_output_labels(
        yolo_result,
        latest_row,
    )

    output_names = {
        "YOLO": detected,
        "Behaviour": behavior,
        "AST": audio_display,
    }

    rows = []
    for modality in ["YOLO", "Behaviour", "AST"]:
        value = round(float(grouped[modality]), 6)
        rows.append(
            {
                "feature": f"{modality} — {output_names[modality]}",
                "modality": modality,
                "output": output_names[modality],
                "shap_value": value,
                "direction": (
                    "supports this risk prediction"
                    if value >= 0
                    else "pushes toward another risk class"
                ),
            }
        )

    fusion_status, fusion_formula = determine_fusion_status(
        yolo_result,
        behavior,
        audio_display,
    )

    return rows, fusion_status, fusion_formula


def make_human_risk_explanation(
    risk_class,
    modality_impacts,
    fusion_status,
    fusion_formula,
):
    ranked = sorted(
        modality_impacts,
        key=lambda item: abs(safe_float(item.get("shap_value", 0.0))),
        reverse=True,
    )
    strongest = ranked[0]["feature"] if ranked else "the combined live inputs"
    return (
        f"{fusion_formula} produced {fusion_status}. "
        f"Therefore, the final risk is {risk_class}. "
        f"The SHAP graph shows that {strongest} had the strongest influence on this risk level."
    )


def predict_live_risk_and_shap(yolo_result):
    input_frame = build_current_risk_feature_frame()
    probabilities, probability_map = get_prediction_probabilities(input_frame)

    latest_row = risk_history[-1] if risk_history else {}
    _, behavior, audio_display = _current_output_labels(yolo_result, latest_row)
    fusion_status, fusion_formula = determine_fusion_status(
        yolo_result,
        behavior,
        audio_display,
    )

    # Final risk follows the project's 11-condition fusion logic.
    # This prevents impossible results such as Baby detected + No Baby risk.
    risk_class = risk_from_fusion_status(fusion_status)
    predicted_class_index = risk_class_index_for_label(risk_class)
    risk_confidence = float(probabilities[predicted_class_index])

    modality_impacts, _, _ = explain_current_prediction(
        input_frame,
        predicted_class_index,
        yolo_result,
    )
    explanation = make_human_risk_explanation(
        risk_class,
        modality_impacts,
        fusion_status,
        fusion_formula,
    )

    return {
        "risk_class": risk_class,
        "risk_confidence": round(risk_confidence, 6),
        "risk_probabilities": probability_map,
        "risk_explanation": explanation,
        "shap_features": modality_impacts,
        "fusion_status": fusion_status,
        "fusion_formula": fusion_formula,
    }


def initialize_live_csv():
    """Create the live history CSV with a stable header if it does not exist."""
    if LIVE_CSV_PATH.exists() and LIVE_CSV_PATH.stat().st_size > 0:
        return

    fieldnames = [
        "sample_id",
        "detected_class",
        "baby_detected",
        "empty_seat_detected",
        "yolo_confidence",
        "lstm_behavior",
        "behavior_confidence",
        "audio_class",
        "audio_confidence",
        "cry_probability",
        "noise_probability",
        "normal_probability",
        "risk_label",
        "risk_confidence",
    ]

    for index in range(1, CSV_SHAP_FEATURE_COUNT + 1):
        fieldnames.extend([
            f"shap_feature_{index}",
            f"shap_value_{index}",
            f"shap_direction_{index}",
        ])

    with LIVE_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def append_live_csv_row(sample_id, yolo_result, behavior, behavior_confidence, audio_result, risk_result):
    """Append one synchronized live prediction without blocking the camera loop."""
    initialize_live_csv()

    row = {
        "sample_id": sample_id,
        "detected_class": yolo_result.get("detected_class", "None"),
        "baby_detected": yolo_result.get("baby_detected", 0),
        "empty_seat_detected": yolo_result.get("empty_seat_detected", 0),
        "yolo_confidence": yolo_result.get("yolo_confidence", 0.0),
        "lstm_behavior": behavior,
        "behavior_confidence": behavior_confidence,
        "audio_class": audio_result.get("audio_class", "Normal"),
        "audio_confidence": audio_result.get("audio_confidence", 0.0),
        "cry_probability": audio_result.get("cry_probability", 0.0),
        "noise_probability": audio_result.get("noise_probability", 0.0),
        "normal_probability": audio_result.get("normal_probability", 0.0),
        "risk_label": risk_result.get("risk_class", "Waiting"),
        "risk_confidence": risk_result.get("risk_confidence", 0.0),
        "fusion_status": risk_result.get("fusion_status", ""),
        "fusion_formula": risk_result.get("fusion_formula", ""),
    }

    shap_features = risk_result.get("shap_features", [])[:CSV_SHAP_FEATURE_COUNT]
    for index in range(1, CSV_SHAP_FEATURE_COUNT + 1):
        item = shap_features[index - 1] if index <= len(shap_features) else {}
        row[f"shap_feature_{index}"] = item.get("feature", "")
        row[f"shap_value_{index}"] = item.get("shap_value", "")
        row[f"shap_direction_{index}"] = item.get("direction", "")

    fieldnames = list(row.keys())
    with LIVE_CSV_PATH.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


def draw_outline_text(frame, text, x, y, font_scale=0.55):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_live_dashboard_overlay(frame, sample_id, yolo_result, pose_detected, movement_score, behavior, behavior_confidence, audio_result, risk_result):
    lines = [
        f"Live Sample: {sample_id}",
        f"YOLO: {yolo_result['detected_class']} | Baby: {yolo_result['baby_count']} | Empty Seat: {yolo_result['empty_seat_count']}",
        f"Pose: {pose_detected} | Movement: {movement_score:.6f}",
        f"Behavior: {behavior} | {behavior_confidence:.1%}",
        f"Audio: {audio_result['audio_class']} | {audio_result['audio_confidence']:.1%}",
        f"Risk: {risk_result['risk_class']} | {risk_result['risk_confidence']:.1%}",
    ]

    for index, line in enumerate(lines):
        draw_outline_text(frame, line, 15, 28 + index * 27)


def encode_latest_frame(frame):
    global latest_encoded_frame
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if success:
        with frame_lock:
            latest_encoded_frame = encoded.tobytes()


def open_camera():
    camera_backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    camera_indices = [CAMERA_INDEX, 1, 2]
    attempted = set()

    for backend in camera_backends:
        for camera_index in camera_indices:
            key = (camera_index, backend)
            if key in attempted:
                continue
            attempted.add(key)

            camera = cv2.VideoCapture(camera_index, backend)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            if camera.isOpened():
                print("Camera opened with index:", camera_index)
                return camera
            camera.release()

    raise RuntimeError("Could not open any camera using indexes 0, 1 or 2.")


def multimodal_processing_worker():
    previous_pose_row = None
    current_behavior = "Collecting Sequence"
    current_behavior_confidence = 0.0
    current_movement_score = 0.0
    sample_id = 0
    last_risk_update_time = 0.0

    current_risk_result = {
        "risk_class": "Waiting",
        "risk_confidence": 0.0,
        "risk_probabilities": {},
        "risk_explanation": "Waiting for the first live multimodal observation.",
        "shap_features": [],
        "fusion_status": "Waiting",
        "fusion_formula": "Waiting for live signals",
    }

    camera = None

    try:
        load_all_live_models()
        initialize_live_csv()
        print(f"Live CSV history: {LIVE_CSV_PATH}")

        audio_thread = threading.Thread(
            target=audio_processing_worker,
            daemon=True,
            name="ASTAudioWorker",
        )
        audio_thread.start()

        camera = open_camera()
        update_live_state(
            system_ready=True,
            camera_ok=True,
            message="Live monitoring is running.",
            error="",
        )

        print("\nLive multimodal monitoring started.")
        print(f"Live history CSV: {LIVE_CSV_PATH}")

        while not stop_event.is_set():
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a camera frame.")

            sample_id += 1
            display_frame, yolo_result = process_yolo(frame)
            pose_row, pose_result = extract_pose_row(frame)
            pose_detected = int(pose_row is not None)

            if pose_row is not None:
                current_movement_score = calculate_landmark_movement(previous_pose_row, pose_row)
                movement_history.append(current_movement_score)
                live_lstm_row = create_live_lstm_feature_row(pose_row)
                ordered_lstm_input = prepare_lstm_input_array(live_lstm_row)
                scaled_feature = np.asarray(lstm_scaler.transform(ordered_lstm_input), dtype=np.float32)[0]
                lstm_sequence_buffer.append(scaled_feature)
                previous_pose_row = pose_row.copy()

                if len(lstm_sequence_buffer) == LSTM_SEQUENCE_LENGTH:
                    current_behavior, current_behavior_confidence = predict_lstm_behavior()
                    current_behavior, current_behavior_confidence = correct_behavior_with_movement(
                        current_behavior,
                        current_behavior_confidence,
                    )
                else:
                    current_behavior = "Collecting Sequence"
                    current_behavior_confidence = len(lstm_sequence_buffer) / LSTM_SEQUENCE_LENGTH
            else:
                current_movement_score = 0.0

            draw_pose(display_frame, pose_result)

            with audio_lock:
                current_audio_result = latest_audio_result.copy()

            raw_risk_row = create_raw_risk_row(
                yolo_result=yolo_result,
                pose_detected=pose_detected,
                movement_score=current_movement_score,
                behavior=current_behavior,
                behavior_confidence=current_behavior_confidence,
                audio_result=current_audio_result,
            )
            append_live_risk_row(raw_risk_row)

            current_time = time.time()
            if current_time - last_risk_update_time >= RISK_UPDATE_INTERVAL_SECONDS:
                try:
                    current_risk_result = predict_live_risk_and_shap(yolo_result)
                    append_live_csv_row(
                        sample_id,
                        yolo_result,
                        current_behavior,
                        current_behavior_confidence,
                        current_audio_result,
                        current_risk_result,
                    )
                    last_risk_update_time = current_time
                except Exception as risk_error:
                    error_text = f"Risk/SHAP error: {risk_error}"
                    print(error_text)
                    current_risk_result = {
                        "risk_class": "Error",
                        "risk_confidence": 0.0,
                        "risk_probabilities": {},
                        "risk_explanation": error_text,
                        "shap_features": [],
                        "fusion_status": "Unavailable",
                        "fusion_formula": "Risk explanation unavailable",
                    }

            # Keep the video clean: YOLO boxes and pose landmarks only.
            # Text results are displayed in the responsive phone dashboard.
            encode_latest_frame(display_frame)

            update_live_state(
                system_ready=True,
                message="Live monitoring is running.",
                sample_id=sample_id,
                detected_class=yolo_result["detected_class"],
                baby_detected=yolo_result["baby_detected"],
                empty_seat_detected=yolo_result["empty_seat_detected"],
                baby_count=yolo_result["baby_count"],
                empty_seat_count=yolo_result["empty_seat_count"],
                yolo_confidence=yolo_result["yolo_confidence"],
                bbox_width=yolo_result["bbox_width"],
                bbox_height=yolo_result["bbox_height"],
                bbox_area=yolo_result["bbox_area"],
                pose_detected=pose_detected,
                movement_score=round(current_movement_score, 8),
                lstm_behavior=current_behavior,
                behavior_confidence=round(current_behavior_confidence, 6),
                audio_class=current_audio_result["audio_class"],
                audio_confidence=current_audio_result["audio_confidence"],
                cry_probability=current_audio_result["cry_probability"],
                noise_probability=current_audio_result["noise_probability"],
                normal_probability=current_audio_result["normal_probability"],
                risk_class=current_risk_result["risk_class"],
                risk_confidence=current_risk_result["risk_confidence"],
                risk_probabilities=current_risk_result["risk_probabilities"],
                risk_explanation=current_risk_result["risk_explanation"],
                shap_features=current_risk_result["shap_features"],
                fusion_status=current_risk_result.get("fusion_status", "Waiting"),
                fusion_formula=current_risk_result.get("fusion_formula", "Waiting for live signals"),
                camera_ok=True,
                error="",
            )

    except Exception as error:
        error_text = f"Live processing stopped: {error}"
        print(error_text)
        update_live_state(
            system_ready=False,
            camera_ok=False,
            message="Live processing error.",
            error=error_text,
        )

    finally:
        stop_event.set()
        if camera is not None:
            camera.release()
        if pose_model is not None:
            try:
                pose_model.close()
            except Exception:
                pass
        print("Multimodal processing worker closed.")


print("Part 2 definitions loaded.")
def generate_mjpeg_stream():
    while not stop_event.is_set():
        with frame_lock:
            frame_bytes = latest_encoded_frame

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.03)


# ============================================================
# 26. FLASK ROUTES
# ============================================================

@app.route("/")
def dashboard_home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg_stream(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/api/live_data")
def api_live_data():
    return jsonify(
        copy_live_state()
    )


@app.route("/api/health")
def api_health():
    state = copy_live_state()

    return jsonify(
        {
            "system_ready": state.get(
                "system_ready",
                False,
            ),
            "camera_ok": state.get(
                "camera_ok",
                False,
            ),
            "audio_ok": state.get(
                "audio_ok",
                False,
            ),
            "message": state.get(
                "message",
                "",
            ),
            "error": state.get(
                "error",
                "",
            ),
        }
    )


@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()

    update_live_state(
        system_ready=False,
        message="Stop request received.",
    )

    return jsonify(
        {
            "success": True,
            "message": (
                "The live monitoring worker "
                "is stopping."
            ),
        }
    )


# ============================================================
# 27. START BACKGROUND MULTIMODAL WORKER
# ============================================================

worker_thread = None
worker_start_lock = threading.Lock()


def start_multimodal_worker_once():
    global worker_thread

    with worker_start_lock:
        if (
            worker_thread is not None
            and worker_thread.is_alive()
        ):
            return

        stop_event.clear()

        worker_thread = threading.Thread(
            target=multimodal_processing_worker,
            daemon=True,
            name="MultimodalAIWorker",
        )

        worker_thread.start()

        print(
            "Multimodal background worker started."
        )


# ============================================================
# 28. MAIN STARTUP
# ============================================================

def print_run_information():
    print("\n" + "=" * 72)
    print("LIVE MULTIMODAL XGBOOST + SHAP PHONE DASHBOARD")
    print("=" * 72)
    print("Local computer:")
    print("http://127.0.0.1:5000")
    print()
    print("Phone on the same Wi-Fi:")
    print("http://YOUR_LAPTOP_IP:5000")
    print()
    print("Example:")
    print("http://192.168.0.102:5000")
    print()
    print("For ngrok, open another terminal and run:")
    print("ngrok http 5000")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    print_run_information()

    start_multimodal_worker_once()

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )

    finally:
        stop_event.set()

        if (
            worker_thread is not None
            and worker_thread.is_alive()
        ):
            worker_thread.join(
                timeout=3.0
            )

        print(
            "Flask dashboard closed."
        )