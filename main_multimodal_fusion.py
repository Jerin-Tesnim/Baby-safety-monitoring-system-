import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import csv
import json
import time
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from collections import deque

import cv2
import joblib
import numpy as np
import pandas as pd
import sounddevice as sd
import torch
import mediapipe as mp
import tensorflow as tf
import keras

from ultralytics import YOLO
from transformers import AutoFeatureExtractor, ASTForAudioClassification


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = (
    r"C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat"
    r"\runs\detect\baby_person_empty_seat"
    r"\weights\best.pt"
)

LSTM_MODEL_PATH = (
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_behavior_model.keras"
)

LSTM_SCALER_PATH = (
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_feature_scaler.pkl"
)

LSTM_LABEL_ENCODER_PATH = (
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_label_encoder.pkl"
)

LSTM_FEATURE_COLUMNS_PATH = (
    r"C:\Users\Admin\Desktop\MediaPipe_33_Landmarks"
    r"\lstm_feature_columns.json"
)

AST_MODEL_FOLDER = (
    r"C:\Users\Admin\Desktop\AST_Audio_Project"
    r"\ast_audio_model"
)

OUTPUT_CSV_PATH = (
    r"C:\Users\Admin\Desktop\Multimodal_Fusion"
    r"\final_multimodal_live_data.csv"
)


# ============================================================
# 2. GENERAL SETTINGS
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

YOLO_CONFIDENCE_THRESHOLD = 0.40
SAVE_INTERVAL_SECONDS = 1.0

SEQUENCE_LENGTH = 30
ROLLING_WINDOW = 30
EPSILON = 1e-8

# Behavior thresholds for stable body-landmark movement
INACTIVE_THRESHOLD = 0.0025
LOW_MOVEMENT_THRESHOLD = 0.0060
RESTLESS_MEAN_THRESHOLD = 0.0120
RESTLESS_STD_THRESHOLD = 0.0040
SUDDEN_MOVEMENT_THRESHOLD = 0.0350
BEHAVIOR_WINDOW = 20

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 1.0
AUDIO_CHANNELS = 1

WINDOW_NAME = "Live Multimodal Fusion"
QUIT_KEY = ord("q")


# ============================================================
# 3. TARGET CLASS NAMES
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


# ============================================================
# 4. CSV COLUMNS
# ============================================================

CSV_COLUMNS = [
    "sample_id",
    "detected_class",
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
    "lstm_behavior",
    "behavior_confidence",
    "audio_class",
    "audio_confidence",
    "cry_probability",
    "noise_probability",
    "normal_probability",
    "distress_detected",
    "distress_status",
]


# ============================================================
# 5. SHARED AUDIO STATE
# ============================================================

audio_lock = threading.Lock()
stop_audio_thread = threading.Event()

latest_audio_result = {
    "audio_class": "Waiting",
    "audio_confidence": 0.0,
    "cry_probability": 0.0,
    "noise_probability": 0.0,
    "normal_probability": 0.0,
}


# ============================================================
# 6. BASIC HELPERS
# ============================================================

def check_required_path(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} was not found:\n{path}\n"
        )


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def class_name_matches(class_name, allowed_names):
    normalized_class = normalize_label(class_name)
    normalized_allowed = {
        normalize_label(name)
        for name in allowed_names
    }
    return normalized_class in normalized_allowed


def load_feature_columns():
    with open(
        LSTM_FEATURE_COLUMNS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in [
            "feature_columns",
            "features",
            "columns",
            "input_features",
        ]:
            if key in data:
                return data[key]

    raise ValueError(
        "Could not read feature columns from "
        "lstm_feature_columns.json."
    )


def load_compatible_lstm_model(model_path):
    """Load a Keras 3 LSTM model inside TensorFlow/Keras 2.15.

    The original model was saved with Keras 3 and cannot be deserialized
    directly by Keras 2.15. This function rebuilds the same architecture
    with Keras 2.15 and copies the trained weights from the .keras archive.
    """
    import h5py

    source_path = Path(model_path)

    # First try normal loading in case a native Keras 2 model is supplied.
    try:
        model = keras.models.load_model(
            str(source_path),
            compile=False
        )
        print("LSTM model loaded directly.")
        return model
    except Exception:
        print(
            "Direct LSTM loading failed. "
            "Rebuilding the Keras 3 model for Keras 2.15..."
        )

    if not zipfile.is_zipfile(source_path):
        raise RuntimeError(
            "The LSTM model is not a valid .keras archive: "
            f"{source_path}"
        )

    temporary_folder = Path(
        tempfile.mkdtemp(prefix="lstm_keras3_weights_")
    )

    try:
        with zipfile.ZipFile(source_path, "r") as archive:
            archive.extract("model.weights.h5", temporary_folder)

        weights_path = temporary_folder / "model.weights.h5"

        # Rebuild the exact architecture used by the uploaded Keras 3 model.
        model = keras.Sequential(
            [
                keras.layers.Input(
                    shape=(30, 143),
                    name="input_layer"
                ),
                keras.layers.LSTM(
                    128,
                    return_sequences=True,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    name="lstm"
                ),
                keras.layers.BatchNormalization(
                    momentum=0.99,
                    epsilon=0.001,
                    name="batch_normalization"
                ),
                keras.layers.Dropout(
                    0.30,
                    name="dropout"
                ),
                keras.layers.LSTM(
                    64,
                    return_sequences=True,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    name="lstm_1"
                ),
                keras.layers.BatchNormalization(
                    momentum=0.99,
                    epsilon=0.001,
                    name="batch_normalization_1"
                ),
                keras.layers.Dropout(
                    0.30,
                    name="dropout_1"
                ),
                keras.layers.LSTM(
                    32,
                    return_sequences=False,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    name="lstm_2"
                ),
                keras.layers.Dropout(
                    0.25,
                    name="dropout_2"
                ),
                keras.layers.Dense(
                    64,
                    activation="relu",
                    name="dense"
                ),
                keras.layers.Dropout(
                    0.25,
                    name="dropout_3"
                ),
                keras.layers.Dense(
                    5,
                    activation="softmax",
                    name="dense_1"
                ),
            ],
            name="sequential"
        )

        # Build all variables before assigning weights.
        model(np.zeros((1, 30, 143), dtype=np.float32), training=False)

        with h5py.File(weights_path, "r") as weights_file:
            layers_group = weights_file["layers"]

            def read_variables(group_path):
                variable_group = layers_group[group_path]["vars"]
                variable_keys = sorted(
                    variable_group.keys(),
                    key=lambda value: int(value)
                )
                return [
                    np.asarray(variable_group[key])
                    for key in variable_keys
                ]

            weight_mapping = {
                "lstm": "lstm/cell",
                "batch_normalization": "batch_normalization",
                "lstm_1": "lstm_1/cell",
                "batch_normalization_1": "batch_normalization_1",
                "lstm_2": "lstm_2/cell",
                "dense": "dense",
                "dense_1": "dense_1",
            }

            for layer_name, archive_group in weight_mapping.items():
                layer = model.get_layer(layer_name)
                archived_weights = read_variables(archive_group)
                expected_weights = layer.get_weights()

                if len(archived_weights) != len(expected_weights):
                    raise RuntimeError(
                        f"Weight count mismatch for {layer_name}: "
                        f"expected {len(expected_weights)}, "
                        f"found {len(archived_weights)}."
                    )

                for index, (expected, archived) in enumerate(
                    zip(expected_weights, archived_weights)
                ):
                    if expected.shape != archived.shape:
                        raise RuntimeError(
                            f"Weight shape mismatch for {layer_name} "
                            f"at index {index}: expected {expected.shape}, "
                            f"found {archived.shape}."
                        )

                layer.set_weights(archived_weights)

        print("Keras 3 LSTM weights loaded successfully in Keras 2.15.")
        print("LSTM input shape:", model.input_shape)
        print("LSTM output shape:", model.output_shape)
        return model

    except Exception as error:
        raise RuntimeError(
            "Keras 3 to Keras 2.15 LSTM conversion failed: "
            f"{error}"
        ) from error

    finally:
        shutil.rmtree(temporary_folder, ignore_errors=True)


# ============================================================
# 7. CSV HELPERS
# ============================================================

def prepare_output_csv():
    os.makedirs(
        os.path.dirname(OUTPUT_CSV_PATH),
        exist_ok=True
    )

    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(
            OUTPUT_CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )
            writer.writeheader()


def get_next_sample_id():
    if not os.path.exists(OUTPUT_CSV_PATH):
        return 1

    last_id = 0

    try:
        with open(
            OUTPUT_CSV_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                try:
                    last_id = max(
                        last_id,
                        int(row["sample_id"])
                    )
                except Exception:
                    continue
    except Exception:
        return 1

    return last_id + 1


def save_row(row):
    with open(
        OUTPUT_CSV_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )
        writer.writerow(row)


# ============================================================
# 8. YOLO PROCESSING
# ============================================================

def process_yolo(yolo_model, frame):
    results = yolo_model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]
    display_frame = frame.copy()

    baby_count = 0
    empty_seat_count = 0

    selected_confidence = 0.0
    selected_bbox_width = 0.0
    selected_bbox_height = 0.0
    selected_bbox_area = 0.0

    detected_labels = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = str(yolo_model.names[class_id])

            is_baby = class_name_matches(
                class_name,
                BABY_CLASS_NAMES
            )

            is_empty_seat = class_name_matches(
                class_name,
                EMPTY_SEAT_CLASS_NAMES
            )

            if not is_baby and not is_empty_seat:
                continue

            if is_baby:
                baby_count += 1
                display_name = "Baby"
            else:
                empty_seat_count += 1
                display_name = "Empty Seat"

            detected_labels.append(display_name)

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            bbox_width = max(0, int(x2 - x1))
            bbox_height = max(0, int(y2 - y1))
            bbox_area = bbox_width * bbox_height

            if confidence > selected_confidence:
                selected_confidence = confidence
                selected_bbox_width = bbox_width
                selected_bbox_height = bbox_height
                selected_bbox_area = bbox_area

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label_text = (
                f"{display_name} {confidence:.2f}"
            )

            cv2.putText(
                display_frame,
                label_text,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    detected_class = (
        ",".join(sorted(set(detected_labels)))
        if detected_labels
        else "None"
    )

    features = {
        "detected_class": detected_class,
        "baby_detected": int(baby_count > 0),
        "empty_seat_detected": int(
            empty_seat_count > 0
        ),
        "baby_count": baby_count,
        "empty_seat_count": empty_seat_count,
        "yolo_confidence": round(
            selected_confidence,
            6
        ),
        "bbox_width": selected_bbox_width,
        "bbox_height": selected_bbox_height,
        "bbox_area": selected_bbox_area,
    }

    return display_frame, features


# ============================================================
# 9. MEDIAPIPE PROCESSING
# ============================================================

def extract_pose_row(frame, pose_model):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb.flags.writeable = False
    result = pose_model.process(rgb)
    rgb.flags.writeable = True

    if result.pose_landmarks is None:
        return None, result

    row = {}

    for index, landmark in enumerate(
        result.pose_landmarks.landmark
    ):
        row[f"landmark_{index}_x"] = float(
            landmark.x
        )
        row[f"landmark_{index}_y"] = float(
            landmark.y
        )
        row[f"landmark_{index}_z"] = float(
            landmark.z
        )
        row[
            f"landmark_{index}_visibility"
        ] = float(landmark.visibility)

    row["pose_detected"] = 1

    return row, result


def draw_pose(frame, pose_result):
    if (
        pose_result is not None
        and pose_result.pose_landmarks is not None
    ):
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )


def calculate_landmark_movement(
    previous_pose_row,
    current_pose_row
):
    if (
        previous_pose_row is None
        or current_pose_row is None
    ):
        return 0.0

    # Stable body landmarks only. Face landmarks are excluded because
    # small camera noise around the eyes and mouth creates false movement.
    stable_landmarks = [
        11, 12, 13, 14, 15, 16,
        23, 24, 25, 26, 27, 28
    ]

    landmark_movements = []

    for index in stable_landmarks:
        previous_visibility = float(
            previous_pose_row.get(
                f"landmark_{index}_visibility",
                0.0
            )
        )

        current_visibility = float(
            current_pose_row.get(
                f"landmark_{index}_visibility",
                0.0
            )
        )

        if min(previous_visibility, current_visibility) < 0.50:
            continue

        previous_x = float(
            previous_pose_row.get(f"landmark_{index}_x", 0.0)
        )
        previous_y = float(
            previous_pose_row.get(f"landmark_{index}_y", 0.0)
        )
        current_x = float(
            current_pose_row.get(f"landmark_{index}_x", 0.0)
        )
        current_y = float(
            current_pose_row.get(f"landmark_{index}_y", 0.0)
        )

        distance = np.sqrt(
            (current_x - previous_x) ** 2
            + (current_y - previous_y) ** 2
        )
        landmark_movements.append(float(distance))

    if not landmark_movements:
        return 0.0

    # Median is more resistant to one unstable landmark than mean.
    return float(np.median(landmark_movements))


# ============================================================
# 10. EXACT LSTM FEATURE ENGINEERING
# ============================================================

def create_live_feature_row(
    pose_row,
    movement_history
):
    row = {}

    if pose_row is not None:
        row.update(pose_row)

    movement_values = np.asarray(
        movement_history,
        dtype=np.float32
    )

    if len(movement_values) == 0:
        movement_values = np.asarray(
            [0.0],
            dtype=np.float32
        )

    movement_score = float(
        movement_values[-1]
    )

    previous_movement = (
        float(movement_values[-2])
        if len(movement_values) >= 2
        else movement_score
    )

    movement_change = abs(
        movement_score - previous_movement
    )

    if len(movement_values) >= 3:
        previous_change = abs(
            float(movement_values[-2])
            - float(movement_values[-3])
        )
    else:
        previous_change = 0.0

    movement_acceleration = abs(
        movement_change - previous_change
    )

    movement_velocity = (
        movement_score - previous_movement
    )

    rolling_values = movement_values[
        -ROLLING_WINDOW:
    ]

    rolling_mean = float(
        np.mean(rolling_values)
    )

    rolling_std = float(
        np.std(rolling_values, ddof=1)
    ) if len(rolling_values) > 1 else 0.0

    rolling_min = float(
        np.min(rolling_values)
    )

    rolling_max = float(
        np.max(rolling_values)
    )

    movement_range = (
        rolling_max - rolling_min
    )

    movement_ratio = (
        movement_score
        / (rolling_mean + EPSILON)
    )

    row["movement_score"] = movement_score
    row["movement_change"] = movement_change
    row[
        "movement_acceleration"
    ] = movement_acceleration
    row["movement_velocity"] = movement_velocity

    row["rolling_mean"] = rolling_mean
    row["rolling_std"] = rolling_std
    row["rolling_min"] = rolling_min
    row["rolling_max"] = rolling_max
    row["movement_range"] = movement_range
    row["movement_ratio"] = movement_ratio

    row[
        "movement_rolling_mean"
    ] = rolling_mean

    row[
        "movement_rolling_std"
    ] = rolling_std

    row[
        "movement_rolling_min"
    ] = rolling_min

    row[
        "movement_rolling_max"
    ] = rolling_max

    row["rolling_range"] = movement_range

    row["movement_mean_30"] = rolling_mean
    row["movement_std_30"] = rolling_std
    row["movement_min_30"] = rolling_min
    row["movement_max_30"] = rolling_max
    row["movement_range_30"] = movement_range

    return row


def prepare_ordered_feature_array(
    live_feature_row,
    feature_columns
):
    ordered_values = []

    for column in feature_columns:
        value = live_feature_row.get(
            column,
            0.0
        )

        try:
            value = float(value)
        except Exception:
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        ordered_values.append(value)

    return np.asarray(
        ordered_values,
        dtype=np.float32
    ).reshape(1, -1)


def predict_lstm_behavior(
    lstm_model,
    lstm_scaler,
    label_encoder,
    sequence_buffer
):
    sequence_array = np.asarray(
        sequence_buffer,
        dtype=np.float32
    )

    sequence_array = np.expand_dims(
        sequence_array,
        axis=0
    )

    probabilities = lstm_model.predict(
        sequence_array,
        verbose=0
    )

    probabilities = np.asarray(
        probabilities
    )[0]

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    if hasattr(label_encoder, "classes_"):
        behavior = str(
            label_encoder.classes_[class_index]
        )
    else:
        behavior = (
            EXPECTED_BEHAVIOR_CLASSES[
                class_index
            ]
            if class_index
            < len(EXPECTED_BEHAVIOR_CLASSES)
            else f"Class_{class_index}"
        )

    return behavior, confidence


def correct_behavior_with_movement(
    lstm_behavior,
    lstm_confidence,
    movement_history
):
    recent_values = np.asarray(
        list(movement_history)[-BEHAVIOR_WINDOW:],
        dtype=np.float32
    )

    if len(recent_values) < 5:
        return lstm_behavior, lstm_confidence

    average_movement = float(np.mean(recent_values))
    movement_std = float(np.std(recent_values))
    maximum_movement = float(np.max(recent_values))

    if maximum_movement >= SUDDEN_MOVEMENT_THRESHOLD:
        return "Sudden_Movement", 0.95

    if (
        average_movement >= RESTLESS_MEAN_THRESHOLD
        and movement_std >= RESTLESS_STD_THRESHOLD
    ):
        return "Restless_Movement", 0.90

    if average_movement < INACTIVE_THRESHOLD:
        return "Inactive", 0.95

    if average_movement < LOW_MOVEMENT_THRESHOLD:
        return "Low_Movement", 0.90

    # In the middle range, keep the trained LSTM decision unless it is
    # stuck on an incompatible extreme class.
    normalized_lstm = normalize_label(lstm_behavior)

    if normalized_lstm in {
        "inactive",
        "low movement",
        "restless movement",
        "sudden movement",
        "normal movement",
    }:
        return lstm_behavior, lstm_confidence

    return "Normal_Movement", max(float(lstm_confidence), 0.70)


# ============================================================
# 11. AST AUDIO PROCESSING
# ============================================================

def record_audio():
    sample_count = int(
        AUDIO_SAMPLE_RATE
        * AUDIO_DURATION_SECONDS
    )

    audio = sd.rec(
        sample_count,
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="float32"
    )

    sd.wait()

    return np.squeeze(
        audio
    ).astype(np.float32)


def get_probability_by_keyword(
    probability_map,
    keywords
):
    for label, probability in (
        probability_map.items()
    ):
        normalized_label = normalize_label(label)

        for keyword in keywords:
            if normalize_label(keyword) in normalized_label:
                return float(probability)

    return 0.0


def standardize_audio_class(label):
    normalized = normalize_label(label)

    if "cry" in normalized:
        return "Cry"

    if "noise" in normalized:
        return "Noise"

    if "normal" in normalized:
        return "Normal"

    return str(label)


def predict_ast(
    audio,
    feature_extractor,
    ast_model,
    device
):
    inputs = feature_extractor(
        audio,
        sampling_rate=AUDIO_SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = inputs[
        "input_values"
    ].to(device)

    with torch.no_grad():
        output = ast_model(
            input_values=input_values
        )

        probabilities = torch.softmax(
            output.logits,
            dim=-1
        )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    id2label = ast_model.config.id2label

    predicted_label = str(
        id2label.get(
            predicted_index,
            predicted_index
        )
    )

    probability_map = {}

    for index, probability in enumerate(
        probabilities
    ):
        label = str(
            id2label.get(index, index)
        )

        probability_map[label] = float(
            probability
        )

    return {
        "audio_class": standardize_audio_class(
            predicted_label
        ),
        "audio_confidence": round(
            confidence,
            6
        ),
        "cry_probability": round(
            get_probability_by_keyword(
                probability_map,
                ["cry", "baby cry"]
            ),
            6
        ),
        "noise_probability": round(
            get_probability_by_keyword(
                probability_map,
                ["noise"]
            ),
            6
        ),
        "normal_probability": round(
            get_probability_by_keyword(
                probability_map,
                ["normal"]
            ),
            6
        ),
    }


def audio_worker(
    feature_extractor,
    ast_model,
    device
):
    global latest_audio_result

    while not stop_audio_thread.is_set():
        try:
            audio = record_audio()

            result = predict_ast(
                audio,
                feature_extractor,
                ast_model,
                device
            )

            with audio_lock:
                latest_audio_result = result.copy()

        except Exception as error:
            print("AST audio error:", error)

            with audio_lock:
                latest_audio_result = {
                    "audio_class": "Audio Error",
                    "audio_confidence": 0.0,
                    "cry_probability": 0.0,
                    "noise_probability": 0.0,
                    "normal_probability": 0.0,
                }

            time.sleep(1.0)


# ============================================================
# 12. RULE-BASED DISTRESS STATUS
# ============================================================

def create_distress_status(
    baby_detected,
    behavior,
    audio_class
):
    if int(baby_detected) == 0:
        return 0, "No Baby Detected"

    behavior_name = normalize_label(
        behavior
    )

    audio_name = normalize_label(
        audio_class
    )

    cry_detected = "cry" in audio_name

    if (
        cry_detected
        and behavior_name
        == "restless movement"
    ):
        return 1, "Distressed Movement"

    if (
        cry_detected
        and behavior_name
        == "sudden movement"
    ):
        return 1, "High Distress"

    if (
        cry_detected
        and behavior_name
        == "inactive"
    ):
        return 1, "Critical Inactivity"

    if (
        cry_detected
        and behavior_name
        == "low movement"
    ):
        return 1, "Possible Distress"

    if (
        cry_detected
        and behavior_name
        == "normal movement"
    ):
        return 1, "Crying with Normal Movement"

    if behavior_name == "restless movement":
        return 0, "Restless Movement"

    if behavior_name == "sudden movement":
        return 0, "Sudden Movement"

    if behavior_name == "inactive":
        return 0, "Inactive Condition"

    if behavior_name == "low movement":
        return 0, "Low Movement Condition"

    return 0, "Normal Condition"


# ============================================================
# 13. DISPLAY
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    font_scale=0.55
):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


def draw_dashboard_text(
    frame,
    sample_id,
    saved_samples,
    yolo_result,
    pose_detected,
    movement_score,
    behavior,
    behavior_confidence,
    audio_result,
    distress_status
):
    lines = [
        f"Next Sample ID: {sample_id}",
        f"Saved Samples: {saved_samples}",
        (
            f"YOLO: {yolo_result['detected_class']} | "
            f"Baby: {yolo_result['baby_count']} | "
            f"Empty Seat: {yolo_result['empty_seat_count']}"
        ),
        (
            f"Pose: {pose_detected} | "
            f"Movement: {movement_score:.6f}"
        ),
        (
            f"Behavior: {behavior} | "
            f"Confidence: {behavior_confidence:.2%}"
        ),
        (
            f"Audio: {audio_result['audio_class']} | "
            f"Confidence: {audio_result['audio_confidence']:.2%}"
        ),
        f"Fusion Status: {distress_status}",
        "Press Q to stop",
    ]

    start_y = 28

    for index, line in enumerate(lines):
        draw_text(
            frame,
            line,
            15,
            start_y + index * 27
        )


# ============================================================
# 14. MAIN PROGRAM
# ============================================================

def main():
    print("=" * 72)
    print("LIVE MULTIMODAL FUSION")
    print("=" * 72)

    required_items = [
        (YOLO_MODEL_PATH, "YOLO model"),
        (LSTM_MODEL_PATH, "LSTM model"),
        (LSTM_SCALER_PATH, "LSTM scaler"),
        (
            LSTM_LABEL_ENCODER_PATH,
            "LSTM label encoder"
        ),
        (
            LSTM_FEATURE_COLUMNS_PATH,
            "LSTM feature columns"
        ),
        (AST_MODEL_FOLDER, "AST model folder"),
    ]

    for path, name in required_items:
        check_required_path(path, name)

    prepare_output_csv()

    sample_id = get_next_sample_id()
    starting_sample_id = sample_id
    saved_samples = 0

    print("\nLoading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO classes:", yolo_model.names)

    print("\nLoading LSTM files...")
    lstm_model = load_compatible_lstm_model(
        LSTM_MODEL_PATH
    )

    lstm_scaler = joblib.load(
        LSTM_SCALER_PATH
    )

    label_encoder = joblib.load(
        LSTM_LABEL_ENCODER_PATH
    )

    feature_columns = load_feature_columns()

    print(
        "LSTM input feature count:",
        len(feature_columns)
    )

    print(
        "LSTM classes:",
        list(label_encoder.classes_)
    )

    model_input_shape = lstm_model.input_shape

    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    model_sequence_length = int(
        model_input_shape[1]
    )

    model_feature_count = int(
        model_input_shape[2]
    )

    if model_sequence_length != SEQUENCE_LENGTH:
        print(
            f"Using model sequence length: "
            f"{model_sequence_length}"
        )

    if model_feature_count != len(
        feature_columns
    ):
        raise ValueError(
            "\nLSTM feature mismatch.\n"
            f"Model expects: {model_feature_count}\n"
            f"JSON contains: {len(feature_columns)}"
        )

    sequence_buffer = deque(
        maxlen=model_sequence_length
    )

    movement_history = deque(
        maxlen=ROLLING_WINDOW
    )

    print("\nLoading MediaPipe...")
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    print("\nLoading AST model...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    ast_feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            AST_MODEL_FOLDER
        )
    )

    ast_model = (
        ASTForAudioClassification.from_pretrained(
            AST_MODEL_FOLDER
        )
    )

    ast_model.to(device)
    ast_model.eval()

    print("AST device:", device)
    print(
        "AST classes:",
        ast_model.config.id2label
    )

    ast_thread = threading.Thread(
        target=audio_worker,
        args=(
            ast_feature_extractor,
            ast_model,
            device
        ),
        daemon=True
    )

    ast_thread.start()

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not camera.isOpened():
        stop_audio_thread.set()
        raise RuntimeError(
            "Could not open the camera."
        )

    previous_pose_row = None

    current_behavior = "Collecting Sequence"
    current_behavior_confidence = 0.0
    current_movement_score = 0.0

    last_save_time = time.time()

    print("\nSystem is running.")
    print("Press Q inside the camera window to stop.\n")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            display_frame, yolo_result = (
                process_yolo(
                    yolo_model,
                    frame
                )
            )

            pose_row, pose_result = (
                extract_pose_row(
                    frame,
                    pose_model
                )
            )

            pose_detected = int(
                pose_row is not None
            )

            if pose_row is not None:
                current_movement_score = (
                    calculate_landmark_movement(
                        previous_pose_row,
                        pose_row
                    )
                )

                movement_history.append(
                    current_movement_score
                )

                live_feature_row = (
                    create_live_feature_row(
                        pose_row,
                        movement_history
                    )
                )

                ordered_feature_array = (
                    prepare_ordered_feature_array(
                        live_feature_row,
                        feature_columns
                    )
                )

                scaled_feature = (
                    lstm_scaler.transform(
                        ordered_feature_array
                    )
                )

                scaled_feature = np.asarray(
                    scaled_feature,
                    dtype=np.float32
                )[0]

                sequence_buffer.append(
                    scaled_feature
                )

                previous_pose_row = pose_row.copy()

                if (
                    len(sequence_buffer)
                    == model_sequence_length
                ):
                    (
                        current_behavior,
                        current_behavior_confidence
                    ) = predict_lstm_behavior(
                        lstm_model,
                        lstm_scaler,
                        label_encoder,
                        sequence_buffer
                    )

                    (
                        current_behavior,
                        current_behavior_confidence
                    ) = correct_behavior_with_movement(
                        current_behavior,
                        current_behavior_confidence,
                        movement_history
                    )
                else:
                    current_behavior = (
                        "Collecting Sequence"
                    )
                    current_behavior_confidence = 0.0

            else:
                current_movement_score = 0.0

            draw_pose(
                display_frame,
                pose_result
            )

            with audio_lock:
                current_audio_result = (
                    latest_audio_result.copy()
                )

            (
                distress_detected,
                distress_status
            ) = create_distress_status(
                yolo_result["baby_detected"],
                current_behavior,
                current_audio_result[
                    "audio_class"
                ]
            )

            current_time = time.time()

            if (
                current_time - last_save_time
                >= SAVE_INTERVAL_SECONDS
            ):
                row = {
                    "sample_id": sample_id,
                    "detected_class": (
                        yolo_result[
                            "detected_class"
                        ]
                    ),
                    "baby_detected": (
                        yolo_result[
                            "baby_detected"
                        ]
                    ),
                    "empty_seat_detected": (
                        yolo_result[
                            "empty_seat_detected"
                        ]
                    ),
                    "baby_count": (
                        yolo_result[
                            "baby_count"
                        ]
                    ),
                    "empty_seat_count": (
                        yolo_result[
                            "empty_seat_count"
                        ]
                    ),
                    "yolo_confidence": (
                        yolo_result[
                            "yolo_confidence"
                        ]
                    ),
                    "bbox_width": (
                        yolo_result[
                            "bbox_width"
                        ]
                    ),
                    "bbox_height": (
                        yolo_result[
                            "bbox_height"
                        ]
                    ),
                    "bbox_area": (
                        yolo_result[
                            "bbox_area"
                        ]
                    ),
                    "pose_detected": pose_detected,
                    "movement_score": round(
                        current_movement_score,
                        8
                    ),
                    "lstm_behavior": (
                        current_behavior
                    ),
                    "behavior_confidence": round(
                        current_behavior_confidence,
                        6
                    ),
                    "audio_class": (
                        current_audio_result[
                            "audio_class"
                        ]
                    ),
                    "audio_confidence": (
                        current_audio_result[
                            "audio_confidence"
                        ]
                    ),
                    "cry_probability": (
                        current_audio_result[
                            "cry_probability"
                        ]
                    ),
                    "noise_probability": (
                        current_audio_result[
                            "noise_probability"
                        ]
                    ),
                    "normal_probability": (
                        current_audio_result[
                            "normal_probability"
                        ]
                    ),
                    "distress_detected": (
                        distress_detected
                    ),
                    "distress_status": (
                        distress_status
                    ),
                }

                save_row(row)

                print(
                    f"Saved Sample {sample_id} | "
                    f"YOLO: {row['detected_class']} | "
                    f"Behavior: {row['lstm_behavior']} | "
                    f"Audio: {row['audio_class']} | "
                    f"Status: {row['distress_status']}"
                )

                sample_id += 1
                saved_samples += 1
                last_save_time = current_time

            draw_dashboard_text(
                display_frame,
                sample_id,
                saved_samples,
                yolo_result,
                pose_detected,
                current_movement_score,
                current_behavior,
                current_behavior_confidence,
                current_audio_result,
                distress_status
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame
            )

            if (
                cv2.waitKey(1) & 0xFF
            ) == QUIT_KEY:
                break

    finally:
        stop_audio_thread.set()

        camera.release()
        pose_model.close()

        cv2.destroyAllWindows()

        ast_thread.join(timeout=2.0)

    last_saved_id = (
        sample_id - 1
        if saved_samples > 0
        else starting_sample_id - 1
    )

    print("\n" + "=" * 72)
    print("MULTIMODAL DATA COLLECTION FINISHED")
    print("=" * 72)
    print("Starting Sample ID :", starting_sample_id)
    print("Last Sample ID     :", last_saved_id)
    print("New Samples Saved  :", saved_samples)
    print("CSV Saved At       :", OUTPUT_CSV_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()