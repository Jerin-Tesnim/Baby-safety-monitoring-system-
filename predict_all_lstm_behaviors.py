import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(
    BASE_DIR,
    "mediapipe_33_landmarks_output.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "lstm_behavior_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "lstm_feature_scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "lstm_label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "lstm_feature_columns.json"
)

VALID_OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "lstm_behavior_all_valid_rows.csv"
)

FULL_OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "lstm_behavior_full_output.csv"
)

SEQUENCE_LENGTH = 30
ROLLING_WINDOW = 30
EPSILON = 1e-8


# Reduce unnecessary TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_required_files():
    """Check whether all required input files exist."""

    required_files = [
        INPUT_CSV,
        MODEL_PATH,
        SCALER_PATH,
        LABEL_ENCODER_PATH,
        FEATURE_COLUMNS_PATH,
    ]

    missing_files = []

    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("\nERROR: The following required files were not found:\n")

        for file_path in missing_files:
            print(file_path)

        print(
            "\nKeep all files inside:\n"
            f"{BASE_DIR}"
        )

        raise FileNotFoundError(
            "One or more required files are missing."
        )


def load_feature_columns():
    """Load the feature column order used during LSTM training."""

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        feature_columns = data

    elif isinstance(data, dict):
        possible_keys = [
            "feature_columns",
            "features",
            "columns",
            "input_features",
        ]

        feature_columns = None

        for key in possible_keys:
            if key in data:
                feature_columns = data[key]
                break

        if feature_columns is None:
            raise ValueError(
                "Could not find the feature column list inside "
                "lstm_feature_columns.json."
            )

    else:
        raise ValueError(
            "Invalid format inside lstm_feature_columns.json."
        )

    return feature_columns


def find_sample_id_column(df):
    """Find the sample ID column."""

    possible_columns = [
        "sample_id",
        "sampleid",
        "frame_id",
        "frame_number",
        "id",
    ]

    lower_column_map = {
        str(column).lower(): column for column in df.columns
    }

    for possible_column in possible_columns:
        if possible_column in lower_column_map:
            return lower_column_map[possible_column]

    return None


def find_pose_column(df):
    """Find the pose detection status column."""

    possible_columns = [
        "pose_detected",
        "pose_detection",
        "pose_found",
        "detected_pose",
    ]

    lower_column_map = {
        str(column).lower(): column for column in df.columns
    }

    for possible_column in possible_columns:
        if possible_column in lower_column_map:
            return lower_column_map[possible_column]

    return None


def find_movement_column(df):
    """Find the movement score column."""

    possible_columns = [
        "movement_score",
        "movement",
        "motion_score",
        "movement_value",
        "motion_value",
    ]

    lower_column_map = {
        str(column).lower(): column for column in df.columns
    }

    for possible_column in possible_columns:
        if possible_column in lower_column_map:
            return lower_column_map[possible_column]

    return None


def get_landmark_columns(df):
    """
    Identify MediaPipe landmark coordinate columns.

    Expected examples:
    landmark_0_x
    landmark_0_y
    landmark_0_z
    landmark_0_visibility

    The function also supports columns such as:
    x0, y0, z0, visibility0
    """

    excluded_keywords = [
        "sample",
        "frame",
        "pose_detected",
        "movement",
        "behavior",
        "label",
        "confidence",
        "time",
        "date",
    ]

    landmark_columns = []

    for column in df.columns:
        column_lower = str(column).lower()

        if any(
            keyword in column_lower
            for keyword in excluded_keywords
        ):
            continue

        is_coordinate = any(
            coordinate in column_lower
            for coordinate in [
                "_x",
                "_y",
                "_z",
                "visibility",
                "landmark",
            ]
        )

        if is_coordinate:
            landmark_columns.append(column)

    return landmark_columns


def calculate_movement_from_landmarks(df, landmark_columns):
    """
    Calculate a movement score if movement_score is not present.

    Movement is calculated as the mean absolute difference between
    consecutive landmark frames.
    """

    if not landmark_columns:
        raise ValueError(
            "No movement_score column or landmark columns were found."
        )

    landmark_data = (
        df[landmark_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    movement = landmark_data.diff().abs().mean(axis=1)
    movement = movement.fillna(0.0)

    return movement


def add_movement_features(df, movement_column):
    """
    Recreate movement-derived features used for LSTM training.
    """

    movement = (
        pd.to_numeric(df[movement_column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    df["movement_score"] = movement

    # Difference between consecutive movement values
    df["movement_change"] = (
        movement.diff().abs().fillna(0.0)
    )

    # Difference between consecutive movement changes
    df["movement_acceleration"] = (
        df["movement_change"].diff().abs().fillna(0.0)
    )

    # Signed movement velocity
    df["movement_velocity"] = (
        movement.diff().fillna(0.0)
    )

    # Rolling statistics
    rolling_object = movement.rolling(
        window=ROLLING_WINDOW,
        min_periods=1
    )

    df["rolling_mean"] = rolling_object.mean()
    df["rolling_std"] = (
        rolling_object.std().fillna(0.0)
    )
    df["rolling_min"] = rolling_object.min()
    df["rolling_max"] = rolling_object.max()

    df["movement_range"] = (
        df["rolling_max"] - df["rolling_min"]
    )

    df["movement_ratio"] = (
        movement / (df["rolling_mean"] + EPSILON)
    )

    # Alternative names that may exist in the saved feature list
    df["movement_rolling_mean"] = df["rolling_mean"]
    df["movement_rolling_std"] = df["rolling_std"]
    df["movement_rolling_min"] = df["rolling_min"]
    df["movement_rolling_max"] = df["rolling_max"]
    df["rolling_range"] = df["movement_range"]

    df["movement_mean_30"] = df["rolling_mean"]
    df["movement_std_30"] = df["rolling_std"]
    df["movement_min_30"] = df["rolling_min"]
    df["movement_max_30"] = df["rolling_max"]
    df["movement_range_30"] = df["movement_range"]

    return df


def prepare_feature_dataframe(df, feature_columns):
    """
    Prepare features in exactly the same order as training.
    """

    working_df = df.copy()

    movement_column = find_movement_column(working_df)
    landmark_columns = get_landmark_columns(working_df)

    if movement_column is None:
        print(
            "\nmovement_score column was not found."
            "\nCalculating movement from landmark differences..."
        )

        working_df["movement_score"] = (
            calculate_movement_from_landmarks(
                working_df,
                landmark_columns
            )
        )

        movement_column = "movement_score"

    working_df = add_movement_features(
        working_df,
        movement_column
    )

    # Convert existing required features to numeric
    for column in feature_columns:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(
                working_df[column],
                errors="coerce"
            )

    missing_features = [
        column
        for column in feature_columns
        if column not in working_df.columns
    ]

    if missing_features:
        print("\nWARNING: Some saved training features were not found:")
        for column in missing_features:
            print(f"  - {column}")

        print(
            "\nThese missing features will be filled with 0.0."
        )

        for column in missing_features:
            working_df[column] = 0.0

    feature_df = working_df[feature_columns].copy()

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_df = feature_df.ffill()
    feature_df = feature_df.bfill()
    feature_df = feature_df.fillna(0.0)

    feature_df = feature_df.astype(np.float32)

    return working_df, feature_df, missing_features


def create_sequences(feature_array, sequence_length):
    """
    Create overlapping sequences.

    For N valid rows and sequence length 30:
    sequence count = N - 30 + 1
    """

    total_rows = len(feature_array)

    if total_rows < sequence_length:
        raise ValueError(
            f"At least {sequence_length} valid rows are required. "
            f"Only {total_rows} valid rows were found."
        )

    sequences = []

    for end_index in range(
        sequence_length - 1,
        total_rows
    ):
        start_index = end_index - sequence_length + 1

        sequence = feature_array[
            start_index:end_index + 1
        ]

        sequences.append(sequence)

    return np.asarray(sequences, dtype=np.float32)


def get_class_names(label_encoder, model_output_count):
    """Get behavior class names from the saved label encoder."""

    if hasattr(label_encoder, "classes_"):
        classes = [
            str(class_name)
            for class_name in label_encoder.classes_
        ]
    else:
        classes = [
            str(index)
            for index in range(model_output_count)
        ]

    if len(classes) != model_output_count:
        raise ValueError(
            "The number of LabelEncoder classes does not match "
            "the LSTM model output."
        )

    return classes


def create_safe_probability_column_name(class_name):
    """Create a safe CSV probability column name."""

    safe_name = (
        str(class_name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"{safe_name}_probability"


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 64)
    print("FULL LSTM BEHAVIOR PREDICTION")
    print("=" * 64)

    check_required_files()

    # --------------------------------------------------------
    # Load input files
    # --------------------------------------------------------

    print("\nLoading MediaPipe CSV...")
    original_df = pd.read_csv(INPUT_CSV)

    print(f"Original CSV rows    : {len(original_df)}")
    print(f"Original CSV columns : {len(original_df.columns)}")

    print("\nLoading saved LSTM files...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = load_feature_columns()

    print(f"LSTM input features  : {len(feature_columns)}")
    print(f"Sequence length      : {SEQUENCE_LENGTH}")

    # --------------------------------------------------------
    # Create row identity
    # --------------------------------------------------------

    sample_id_column = find_sample_id_column(original_df)
    pose_column = find_pose_column(original_df)

    original_df["_original_row_index"] = np.arange(
        len(original_df)
    )

    if sample_id_column is None:
        print(
            "\nsample_id column was not found."
            "\nA new sample_id column will be created."
        )

        original_df["sample_id"] = np.arange(
            1,
            len(original_df) + 1
        )

        sample_id_column = "sample_id"

    # --------------------------------------------------------
    # Select valid pose rows
    # --------------------------------------------------------

    if pose_column is not None:
        pose_values = (
            pd.to_numeric(
                original_df[pose_column],
                errors="coerce"
            )
            .fillna(0)
        )

        valid_pose_mask = pose_values == 1

    else:
        print(
            "\npose_detected column was not found."
            "\nValid rows will be selected using landmark data."
        )

        landmark_columns = get_landmark_columns(original_df)

        if not landmark_columns:
            raise ValueError(
                "Could not identify valid pose rows because neither "
                "pose_detected nor landmark columns were found."
            )

        numeric_landmarks = original_df[
            landmark_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        valid_pose_mask = (
            numeric_landmarks.notna().any(axis=1)
        )

    valid_df = (
        original_df.loc[valid_pose_mask]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Valid pose rows      : {len(valid_df)}")
    print(
        f"Invalid pose rows    : "
        f"{len(original_df) - len(valid_df)}"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    prepared_df, feature_df, missing_features = (
        prepare_feature_dataframe(
            valid_df,
            feature_columns
        )
    )

    print(
        f"Prepared feature shape: {feature_df.shape}"
    )

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    print("\nScaling features...")

    scaled_features = scaler.transform(feature_df)

    scaled_features = np.asarray(
        scaled_features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Create all sequences
    # --------------------------------------------------------

    print("Creating 30-frame LSTM sequences...")

    sequences = create_sequences(
        scaled_features,
        SEQUENCE_LENGTH
    )

    print(f"Sequence shape       : {sequences.shape}")
    print(f"Prediction sequences : {len(sequences)}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nPredicting behaviors for all valid rows...")

    probabilities = model.predict(
        sequences,
        batch_size=32,
        verbose=1
    )

    probabilities = np.asarray(probabilities)

    class_names = get_class_names(
        label_encoder,
        probabilities.shape[1]
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_behaviors = np.asarray(
        class_names,
        dtype=object
    )[predicted_indices]

    behavior_confidences = np.max(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Fill first 29 rows
    # --------------------------------------------------------
    #
    # The first complete sequence ends at row 30.
    # Therefore, model predictions naturally start from valid row 30.
    #
    # To keep one output for every valid MediaPipe row, the prediction
    # of the first complete sequence is copied to the first 29 rows.
    # --------------------------------------------------------

    prefix_count = SEQUENCE_LENGTH - 1

    first_behavior = predicted_behaviors[0]
    first_confidence = behavior_confidences[0]
    first_probability = probabilities[0]

    prefix_behaviors = np.full(
        prefix_count,
        first_behavior,
        dtype=object
    )

    prefix_confidences = np.full(
        prefix_count,
        first_confidence,
        dtype=np.float32
    )

    prefix_probabilities = np.tile(
        first_probability,
        (prefix_count, 1)
    )

    all_valid_behaviors = np.concatenate([
        prefix_behaviors,
        predicted_behaviors
    ])

    all_valid_confidences = np.concatenate([
        prefix_confidences,
        behavior_confidences
    ])

    all_valid_probabilities = np.vstack([
        prefix_probabilities,
        probabilities
    ])

    if len(all_valid_behaviors) != len(valid_df):
        raise RuntimeError(
            "Prediction row count does not match valid pose row count."
        )

    # --------------------------------------------------------
    # Create valid-row output
    # --------------------------------------------------------

    valid_output = pd.DataFrame({
        "sample_id": valid_df[sample_id_column].values,
        "pose_detected": 1,
        "lstm_behavior": all_valid_behaviors,
        "behavior_confidence": np.round(
            all_valid_confidences,
            6
        ),
    })

    movement_column = find_movement_column(prepared_df)

    if movement_column is not None:
        valid_output.insert(
            2,
            "movement_score",
            pd.to_numeric(
                prepared_df[movement_column],
                errors="coerce"
            )
            .fillna(0.0)
            .values
        )

    for class_index, class_name in enumerate(class_names):
        probability_column = (
            create_safe_probability_column_name(
                class_name
            )
        )

        valid_output[probability_column] = np.round(
            all_valid_probabilities[:, class_index],
            6
        )

    valid_output.to_csv(
        VALID_OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Create full original-row output
    # --------------------------------------------------------

    full_output = pd.DataFrame({
        "sample_id": original_df[sample_id_column].values,
        "pose_detected": 0,
        "lstm_behavior": "No_Pose",
        "behavior_confidence": 0.0,
    })

    if movement_column is not None:
        original_movement_column = find_movement_column(
            original_df
        )

        if original_movement_column is not None:
            movement_values = pd.to_numeric(
                original_df[original_movement_column],
                errors="coerce"
            ).fillna(0.0)

        else:
            movement_values = pd.Series(
                np.zeros(len(original_df))
            )

        full_output.insert(
            2,
            "movement_score",
            movement_values.values
        )

    for class_name in class_names:
        probability_column = (
            create_safe_probability_column_name(
                class_name
            )
        )

        full_output[probability_column] = 0.0

    valid_original_indices = valid_df[
        "_original_row_index"
    ].astype(int).values

    full_output.loc[
        valid_original_indices,
        "pose_detected"
    ] = 1

    full_output.loc[
        valid_original_indices,
        "lstm_behavior"
    ] = all_valid_behaviors

    full_output.loc[
        valid_original_indices,
        "behavior_confidence"
    ] = np.round(
        all_valid_confidences,
        6
    )

    for class_index, class_name in enumerate(class_names):
        probability_column = (
            create_safe_probability_column_name(
                class_name
            )
        )

        full_output.loc[
            valid_original_indices,
            probability_column
        ] = np.round(
            all_valid_probabilities[:, class_index],
            6
        )

    full_output.to_csv(
        FULL_OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    behavior_counts = (
        valid_output["lstm_behavior"]
        .value_counts()
    )

    print("\n" + "=" * 64)
    print("FULL LSTM PREDICTION COMPLETED")
    print("=" * 64)

    print(f"\nOriginal input rows : {len(original_df)}")
    print(f"Valid pose rows     : {len(valid_output)}")
    print(f"Full output rows    : {len(full_output)}")

    print("\nBehavior counts:")

    for behavior_name, count in behavior_counts.items():
        print(f"{behavior_name:<22}: {count}")

    print("\nValid pose output:")
    print(VALID_OUTPUT_CSV)

    print("\nFull original-row output:")
    print(FULL_OUTPUT_CSV)

    if missing_features:
        print(
            "\nWARNING:"
            "\nSome training features were missing and were filled "
            "with zero."
            "\nReview the missing-feature list printed above."
        )
    else:
        print(
            "\nAll saved training features were found successfully."
        )

    print("=" * 64)


if __name__ == "__main__":
    main()