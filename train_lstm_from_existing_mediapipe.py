import os
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "mediapipe_33_landmarks_output.csv"

SEQUENCE_LENGTH = 30
SEQUENCE_STEP = 5

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32

ROLLING_WINDOW = 10

MODEL_FILE = "lstm_behavior_model.keras"
SCALER_FILE = "lstm_feature_scaler.pkl"
LABEL_ENCODER_FILE = "lstm_label_encoder.pkl"
FEATURE_COLUMNS_FILE = "lstm_feature_columns.json"

LABELED_CSV_FILE = "mediapipe_automatic_behavior_labels.csv"
LSTM_OUTPUT_CSV_FILE = "lstm_behavior_output.csv"

CLASSIFICATION_REPORT_FILE = "lstm_classification_report.csv"
CONFUSION_MATRIX_CSV_FILE = "lstm_confusion_matrix.csv"
CONFUSION_MATRIX_IMAGE_FILE = "lstm_confusion_matrix.png"

TRAINING_HISTORY_FILE = "lstm_training_history.csv"
ACCURACY_GRAPH_FILE = "lstm_accuracy_graph.png"
LOSS_GRAPH_FILE = "lstm_loss_graph.png"


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

warnings.filterwarnings("ignore")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"\nCSV file was not found:\n"
            f"{os.path.abspath(CSV_FILE)}\n\n"
            f"Keep {CSV_FILE} in the same folder as this Python file."
        )

    df = pd.read_csv(CSV_FILE)

    print("\n================================================")
    print("MEDIAPIPE DATASET LOADED")
    print("================================================")
    print("CSV file       :", os.path.abspath(CSV_FILE))
    print("Original shape :", df.shape)
    print("Total columns  :", len(df.columns))

    required_columns = [
        "sample_id",
        "pose_detected",
        "movement_score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df["sample_id"] = pd.to_numeric(
        df["sample_id"],
        errors="coerce"
    )

    df["pose_detected"] = pd.to_numeric(
        df["pose_detected"],
        errors="coerce"
    )

    df["movement_score"] = pd.to_numeric(
        df["movement_score"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "sample_id",
            "pose_detected",
            "movement_score"
        ]
    ).copy()

    df = df[
        df["pose_detected"] == 1
    ].copy()

    df = df.sort_values(
        by="sample_id"
    ).reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH * 5:
        raise ValueError(
            "Dataset is too small for LSTM training.\n"
            f"At least {SEQUENCE_LENGTH * 5} valid rows are recommended.\n"
            f"Valid rows found: {len(df)}"
        )

    print("Valid pose rows :", len(df))
    print("================================================")

    return df


# ============================================================
# PREPARE LANDMARK COLUMNS
# ============================================================

def prepare_landmark_columns(df):
    excluded_columns = {
        "sample_id",
        "pose_detected",
        "movement_score",
        "behavior_label"
    }

    possible_landmark_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    landmark_columns = []

    for column in possible_landmark_columns:
        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().sum() > 0:
            df[column] = converted.fillna(0.0)
            landmark_columns.append(column)

    if not landmark_columns:
        raise ValueError(
            "No MediaPipe landmark columns were found."
        )

    print("\nLandmark features found:", len(landmark_columns))

    return df, landmark_columns


# ============================================================
# CREATE MOVEMENT FEATURES
# ============================================================

def create_movement_features(df):
    movement = df["movement_score"].astype(float)

    df["movement_change"] = (
        movement.diff()
        .fillna(0.0)
    )

    df["absolute_movement_change"] = (
        df["movement_change"].abs()
    )

    df["movement_acceleration"] = (
        df["movement_change"]
        .diff()
        .fillna(0.0)
    )

    df["absolute_movement_acceleration"] = (
        df["movement_acceleration"].abs()
    )

    df["rolling_movement_mean"] = (
        movement
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    df["rolling_movement_std"] = (
        movement
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .std()
        .fillna(0.0)
    )

    df["rolling_movement_max"] = (
        movement
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .max()
    )

    df["rolling_movement_min"] = (
        movement
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .min()
    )

    df["movement_range"] = (
        df["rolling_movement_max"]
        - df["rolling_movement_min"]
    )

    df["movement_ratio"] = (
        movement
        / (
            df["rolling_movement_mean"]
            + 1e-8
        )
    )

    return df


# ============================================================
# AUTOMATIC PSEUDO LABELS
# ============================================================

def create_automatic_behavior_labels(df):
    movement = df["movement_score"]
    rolling_mean = df["rolling_movement_mean"]
    rolling_std = df["rolling_movement_std"]
    absolute_change = df["absolute_movement_change"]
    acceleration = df["absolute_movement_acceleration"]
    movement_ratio = df["movement_ratio"]

    movement_q20 = movement.quantile(0.20)
    movement_q40 = movement.quantile(0.40)
    movement_q70 = movement.quantile(0.70)
    movement_q80 = movement.quantile(0.80)

    rolling_mean_q75 = rolling_mean.quantile(0.75)
    rolling_std_q70 = rolling_std.quantile(0.70)

    change_q90 = absolute_change.quantile(0.90)
    acceleration_q90 = acceleration.quantile(0.90)
    ratio_q90 = movement_ratio.quantile(0.90)

    print("\n================================================")
    print("AUTOMATIC LABEL THRESHOLDS")
    print("================================================")
    print(f"Movement 20th percentile     : {movement_q20:.8f}")
    print(f"Movement 40th percentile     : {movement_q40:.8f}")
    print(f"Movement 70th percentile     : {movement_q70:.8f}")
    print(f"Movement 80th percentile     : {movement_q80:.8f}")
    print(f"Rolling mean 75th percentile : {rolling_mean_q75:.8f}")
    print(f"Rolling std 70th percentile  : {rolling_std_q70:.8f}")
    print(f"Movement change 90th         : {change_q90:.8f}")
    print(f"Acceleration 90th            : {acceleration_q90:.8f}")
    print(f"Movement ratio 90th          : {ratio_q90:.8f}")
    print("================================================")

    labels = []

    for index, row in df.iterrows():
        current_movement = row["movement_score"]
        current_rolling_mean = row["rolling_movement_mean"]
        current_rolling_std = row["rolling_movement_std"]
        current_change = row["absolute_movement_change"]
        current_acceleration = row[
            "absolute_movement_acceleration"
        ]
        current_ratio = row["movement_ratio"]

        sudden_condition = (
            (
                current_change >= change_q90
                or current_acceleration >= acceleration_q90
                or current_ratio >= ratio_q90
            )
            and current_movement >= movement_q70
        )

        restless_condition = (
            current_rolling_mean >= rolling_mean_q75
            and current_rolling_std >= rolling_std_q70
            and current_movement >= movement_q70
        )

        if sudden_condition:
            label = "Sudden_Movement"

        elif restless_condition:
            label = "Restless_Movement"

        elif current_movement <= movement_q20:
            label = "Inactive"

        elif current_movement <= movement_q40:
            label = "Low_Movement"

        else:
            label = "Normal_Movement"

        labels.append(label)

    df["behavior_label"] = labels

    return df


# ============================================================
# BALANCE CHECK
# ============================================================

def check_behavior_distribution(df):
    expected_classes = [
        "Inactive",
        "Low_Movement",
        "Normal_Movement",
        "Restless_Movement",
        "Sudden_Movement"
    ]

    counts = df["behavior_label"].value_counts()

    print("\n================================================")
    print("AUTOMATIC BEHAVIOR LABEL COUNTS")
    print("================================================")

    for behavior in expected_classes:
        print(
            f"{behavior:<20}: "
            f"{int(counts.get(behavior, 0))}"
        )

    print("================================================")

    missing_classes = [
        behavior
        for behavior in expected_classes
        if counts.get(behavior, 0) < SEQUENCE_LENGTH
    ]

    if missing_classes:
        print("\nWarning:")
        print(
            "These classes have fewer than "
            f"{SEQUENCE_LENGTH} frames:"
        )

        for behavior in missing_classes:
            print("-", behavior)

        print(
            "\nThe program will continue, but the "
            "LSTM result for these classes may be weak."
        )


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(df, landmark_columns):
    derived_feature_columns = [
        "movement_score",
        "movement_change",
        "absolute_movement_change",
        "movement_acceleration",
        "absolute_movement_acceleration",
        "rolling_movement_mean",
        "rolling_movement_std",
        "rolling_movement_max",
        "rolling_movement_min",
        "movement_range",
        "movement_ratio"
    ]

    feature_columns = (
        landmark_columns
        + derived_feature_columns
    )

    feature_columns = list(
        dict.fromkeys(feature_columns)
    )

    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0)

    with open(
        FEATURE_COLUMNS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            feature_columns,
            file,
            indent=4
        )

    print("\nTotal LSTM input features:", len(feature_columns))

    return feature_columns


# ============================================================
# CREATE LSTM SEQUENCES
# ============================================================

def create_sequences(df, feature_columns):
    sequences = []
    sequence_labels = []
    sequence_sample_ids = []

    feature_values = df[
        feature_columns
    ].values.astype(np.float32)

    labels = df[
        "behavior_label"
    ].values

    sample_ids = df[
        "sample_id"
    ].values

    last_start = (
        len(df)
        - SEQUENCE_LENGTH
        + 1
    )

    for start_index in range(
        0,
        last_start,
        SEQUENCE_STEP
    ):
        end_index = (
            start_index
            + SEQUENCE_LENGTH
        )

        sequence = feature_values[
            start_index:end_index
        ]

        sequence_label_values = labels[
            start_index:end_index
        ]

        label_counts = pd.Series(
            sequence_label_values
        ).value_counts()

        majority_label = (
            label_counts.index[0]
        )

        sequences.append(sequence)
        sequence_labels.append(majority_label)
        sequence_sample_ids.append(
            int(sample_ids[end_index - 1])
        )

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    y_text = np.asarray(
        sequence_labels
    )

    sample_ids_array = np.asarray(
        sequence_sample_ids
    )

    if len(X) == 0:
        raise ValueError(
            "No LSTM sequences were created."
        )

    print("\nSequence shape:", X.shape)

    print("\nSequence label counts:")
    print(
        pd.Series(y_text).value_counts()
    )

    return X, y_text, sample_ids_array


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_lstm_model(
    sequence_length,
    number_of_features,
    number_of_classes
):
    model = Sequential([
        Input(
            shape=(
                sequence_length,
                number_of_features
            )
        ),

        LSTM(
            128,
            return_sequences=True
        ),

        BatchNormalization(),

        Dropout(0.30),

        LSTM(
            64,
            return_sequences=True
        ),

        BatchNormalization(),

        Dropout(0.30),

        LSTM(
            32,
            return_sequences=False
        ),

        Dropout(0.25),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.25),

        Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

def save_training_graphs(history):
    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        TRAINING_HISTORY_FILE,
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "LSTM Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ACCURACY_GRAPH_FILE,
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "LSTM Training and Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOSS_GRAPH_FILE,
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n================================================")
    print("LSTM TRAINING FROM EXISTING MEDIAPIPE DATA")
    print("================================================")

    df = load_dataset()

    df, landmark_columns = prepare_landmark_columns(
        df
    )

    df = create_movement_features(
        df
    )

    df = create_automatic_behavior_labels(
        df
    )

    check_behavior_distribution(
        df
    )

    df.to_csv(
        LABELED_CSV_FILE,
        index=False
    )

    print("\nAutomatically labeled CSV saved:")
    print(os.path.abspath(LABELED_CSV_FILE))

    feature_columns = get_feature_columns(
        df,
        landmark_columns
    )

    X, text_labels, sequence_sample_ids = (
        create_sequences(
            df,
            feature_columns
        )
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    encoded_labels = (
        label_encoder.fit_transform(
            text_labels
        )
    )

    class_names = (
        label_encoder.classes_
    )

    print("\nLSTM behavior classes:")
    print(class_names)

    if len(class_names) < 2:
        raise ValueError(
            "Only one behavior class was created. "
            "LSTM requires at least two classes."
        )

    class_counts = np.bincount(
        encoded_labels
    )

    if np.min(class_counts) < 2:
        raise ValueError(
            "At least one behavior has fewer than "
            "2 sequences. More variation is required."
        )

    y = to_categorical(
        encoded_labels,
        num_classes=len(class_names)
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_FILE
    )

    # --------------------------------------------------------
    # Train and test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sample_ids_train,
        sample_ids_test
    ) = train_test_split(
        X,
        y,
        sequence_sample_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=encoded_labels
    )

    # --------------------------------------------------------
    # Scale data using training data only
    # --------------------------------------------------------

    number_of_features = X.shape[2]

    train_flattened = X_train.reshape(
        -1,
        number_of_features
    )

    test_flattened = X_test.reshape(
        -1,
        number_of_features
    )

    scaler = StandardScaler()

    train_scaled_flattened = (
        scaler.fit_transform(
            train_flattened
        )
    )

    test_scaled_flattened = (
        scaler.transform(
            test_flattened
        )
    )

    X_train_scaled = (
        train_scaled_flattened.reshape(
            X_train.shape
        )
    )

    X_test_scaled = (
        test_scaled_flattened.reshape(
            X_test.shape
        )
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print("\nTraining sequences:", len(X_train_scaled))
    print("Testing sequences :", len(X_test_scaled))

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=number_of_features,
        number_of_classes=len(class_names)
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00001,
            verbose=1
        ),

        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=VALIDATION_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.save(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Test model
    # --------------------------------------------------------

    test_loss, test_accuracy = model.evaluate(
        X_test_scaled,
        y_test,
        verbose=0
    )

    prediction_probabilities = model.predict(
        X_test_scaled,
        verbose=0
    )

    predicted_indices = np.argmax(
        prediction_probabilities,
        axis=1
    )

    true_indices = np.argmax(
        y_test,
        axis=1
    )

    predicted_labels = (
        label_encoder.inverse_transform(
            predicted_indices
        )
    )

    true_labels = (
        label_encoder.inverse_transform(
            true_indices
        )
    )

    prediction_confidence = np.max(
        prediction_probabilities,
        axis=1
    )

    final_accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print("\n================================================")
    print("LSTM TEST RESULT")
    print("================================================")
    print(f"Test loss     : {test_loss:.6f}")
    print(
        f"Test accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Final accuracy: "
        f"{final_accuracy * 100:.2f}%"
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report_text)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(
        report_dictionary
    ).transpose().to_csv(
        CLASSIFICATION_REPORT_FILE
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_names
    )

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    ).to_csv(
        CONFUSION_MATRIX_CSV_FILE
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        xticks_rotation=35
    )

    plt.title(
        "LSTM Behavior Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_IMAGE_FILE,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Save prediction output
    # --------------------------------------------------------

    output_df = pd.DataFrame({
        "sample_id": sample_ids_test,
        "true_behavior": true_labels,
        "lstm_behavior": predicted_labels,
        "behavior_confidence": prediction_confidence
    })

    for class_index, class_name in enumerate(
        class_names
    ):
        output_df[
            f"{class_name}_probability"
        ] = prediction_probabilities[
            :,
            class_index
        ]

    output_df = output_df.sort_values(
        by="sample_id"
    ).reset_index(drop=True)

    output_df.to_csv(
        LSTM_OUTPUT_CSV_FILE,
        index=False
    )

    save_training_graphs(
        history
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n================================================")
    print("TRAINING COMPLETED")
    print("================================================")

    print("\nLSTM model:")
    print(os.path.abspath(MODEL_FILE))

    print("\nScaler:")
    print(os.path.abspath(SCALER_FILE))

    print("\nLabel encoder:")
    print(os.path.abspath(LABEL_ENCODER_FILE))

    print("\nFeature columns:")
    print(os.path.abspath(FEATURE_COLUMNS_FILE))

    print("\nAutomatically labeled data:")
    print(os.path.abspath(LABELED_CSV_FILE))

    print("\nLSTM prediction output:")
    print(os.path.abspath(LSTM_OUTPUT_CSV_FILE))

    print("\nClassification report:")
    print(os.path.abspath(
        CLASSIFICATION_REPORT_FILE
    ))

    print("\nConfusion matrix:")
    print(os.path.abspath(
        CONFUSION_MATRIX_IMAGE_FILE
    ))

    print("\nAccuracy graph:")
    print(os.path.abspath(
        ACCURACY_GRAPH_FILE
    ))

    print("\nLoss graph:")
    print(os.path.abspath(
        LOSS_GRAPH_FILE
    ))

    print("================================================")


if __name__ == "__main__":
    main()