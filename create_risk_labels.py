import os
from collections import Counter, deque

import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================
INPUT_CSV = "final_multimodal_live_data.csv"
OUTPUT_CSV = "final_fusion_risk_dataset.csv"
SUMMARY_CSV = "risk_level_summary.csv"


# =========================================================
# CONFIDENCE FILTERING SETTINGS
# =========================================================
YOLO_CONFIDENCE_THRESHOLD = 0.50
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.60
AUDIO_CONFIDENCE_THRESHOLD = 0.60


# =========================================================
# TEMPORAL SMOOTHING SETTINGS
# =========================================================
SMOOTHING_WINDOW = 15
MINIMUM_HISTORY = 8


# =========================================================
# HIGH-RISK SETTINGS
# =========================================================

# Inactive must continue for many samples before becoming High Risk
INACTIVE_HIGH_RISK_STREAK = 45

# Number of Sudden Movement results required in the recent window
SUDDEN_HISTORY_WINDOW = 20
SUDDEN_HIGH_RISK_COUNT = 12

# Cry probability thresholds
STRONG_CRY_PROBABILITY = 0.80
VERY_STRONG_CRY_PROBABILITY = 0.90


# =========================================================
# SAFE CONVERSION FUNCTIONS
# =========================================================
def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


# =========================================================
# NORMALIZATION FUNCTIONS
# =========================================================
def normalize_behavior(value):
    text = str(value).strip()

    valid_behaviors = {
        "Inactive",
        "Low_Movement",
        "Normal_Movement",
        "Restless_Movement",
        "Sudden_Movement",
    }

    if text in valid_behaviors:
        return text

    return "Unknown"


def normalize_audio(value):
    text = str(value).strip()

    valid_audio_classes = {
        "Normal",
        "Noise",
        "Cry",
    }

    if text in valid_audio_classes:
        return text

    return "Unknown"


# =========================================================
# MAJORITY VOTING
# =========================================================
def majority_value(values, default_value):
    valid_values = [
        value
        for value in values
        if value not in [
            "Unknown",
            "Collecting Sequence",
            None,
        ]
    ]

    if len(valid_values) == 0:
        return default_value

    counts = Counter(valid_values)
    return counts.most_common(1)[0][0]


# =========================================================
# RISK CALCULATION
# =========================================================
def calculate_risk(
    baby_detected,
    behavior,
    audio,
    cry_probability,
    inactive_streak,
    recent_sudden_count,
):
    """
    Final classes:
        0 = No_Baby
        1 = Low_Risk
        2 = Medium_Risk
        3 = High_Risk
    """

    # -----------------------------------------------------
    # CLASS 1: NO BABY
    # -----------------------------------------------------
    if baby_detected == 0:
        return (
            0,
            "No_Baby",
            "No baby detected by YOLO",
        )

    # -----------------------------------------------------
    # CLASS 4: HIGH RISK
    # -----------------------------------------------------

    # Cry with inactive behavior
    if audio == "Cry" and behavior == "Inactive":
        return (
            3,
            "High_Risk",
            "Crying detected with inactive behavior",
        )

    # Cry with low movement
    if audio == "Cry" and behavior == "Low_Movement":
        return (
            3,
            "High_Risk",
            "Crying detected with low movement",
        )

    # Cry with restless movement
    if audio == "Cry" and behavior == "Restless_Movement":
        return (
            3,
            "High_Risk",
            "Crying detected with restless movement",
        )

    # Cry with sudden movement
    if audio == "Cry" and behavior == "Sudden_Movement":
        return (
            3,
            "High_Risk",
            "Crying detected with sudden movement",
        )

    # Very high cry probability with abnormal behavior
    if (
        cry_probability >= VERY_STRONG_CRY_PROBABILITY
        and behavior in [
            "Inactive",
            "Low_Movement",
            "Restless_Movement",
            "Sudden_Movement",
        ]
    ):
        return (
            3,
            "High_Risk",
            "Very high cry probability with abnormal behavior",
        )

    # Long-duration inactivity
    if inactive_streak >= INACTIVE_HIGH_RISK_STREAK:
        return (
            3,
            "High_Risk",
            "Inactive behavior continued for a long duration",
        )

    # Repeated sudden movement
    if recent_sudden_count >= SUDDEN_HIGH_RISK_COUNT:
        return (
            3,
            "High_Risk",
            "Repeated sudden movement detected",
        )

    # -----------------------------------------------------
    # CLASS 3: MEDIUM RISK
    # -----------------------------------------------------

    # Cry with normal movement
    if audio == "Cry" and behavior == "Normal_Movement":
        return (
            2,
            "Medium_Risk",
            "Crying detected with normal movement",
        )

    # Strong cry probability
    if cry_probability >= STRONG_CRY_PROBABILITY:
        return (
            2,
            "Medium_Risk",
            "Strong cry probability detected",
        )

    # Abnormal behavior without confirmed crying
    if behavior == "Inactive":
        return (
            2,
            "Medium_Risk",
            "Inactive behavior detected",
        )

    if behavior == "Low_Movement":
        return (
            2,
            "Medium_Risk",
            "Low movement detected",
        )

    if behavior == "Restless_Movement":
        return (
            2,
            "Medium_Risk",
            "Restless movement detected",
        )

    if behavior == "Sudden_Movement":
        return (
            2,
            "Medium_Risk",
            "Sudden movement detected",
        )

    # Cry detected while behavior is uncertain
    if audio == "Cry":
        return (
            2,
            "Medium_Risk",
            "Crying detected but behavior is uncertain",
        )

    # -----------------------------------------------------
    # CLASS 2: LOW RISK
    # -----------------------------------------------------

    if (
        behavior == "Normal_Movement"
        and audio in ["Normal", "Noise"]
    ):
        return (
            1,
            "Low_Risk",
            "Normal movement without confirmed crying",
        )

    # One modality is uncertain
    if behavior == "Unknown" or audio == "Unknown":
        return (
            1,
            "Low_Risk",
            "Baby detected but one modality is uncertain",
        )

    return (
        1,
        "Low_Risk",
        "No strong distress condition detected",
    )


# =========================================================
# MAIN PROGRAM
# =========================================================
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV not found: {INPUT_CSV}")
        return

    print("=" * 70)
    print("MULTIMODAL RISK LABEL GENERATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # -----------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------------------------------
    required_columns = [
        "sample_id",
        "baby_detected",
        "empty_seat_detected",
        "yolo_confidence",
        "pose_detected",
        "lstm_behavior",
        "behavior_confidence",
        "audio_class",
        "audio_confidence",
        "cry_probability",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print()
        print("Error: Missing required columns:")

        for column in missing_columns:
            print(f"- {column}")

        return

    # -----------------------------------------------------
    # REMOVE INCOMPLETE LSTM SEQUENCE ROWS
    # -----------------------------------------------------
    df = df[
        df["lstm_behavior"]
        .astype(str)
        .str.strip()
        != "Collecting Sequence"
    ].copy()

    df = df.reset_index(drop=True)

    print(
        "Rows after removing incomplete sequences: "
        f"{len(df)}"
    )

    # -----------------------------------------------------
    # HISTORY BUFFERS
    # -----------------------------------------------------
    baby_history = deque(maxlen=SMOOTHING_WINDOW)
    behavior_history = deque(maxlen=SMOOTHING_WINDOW)
    audio_history = deque(maxlen=SMOOTHING_WINDOW)
    cry_probability_history = deque(maxlen=SMOOTHING_WINDOW)

    sudden_history = deque(maxlen=SUDDEN_HISTORY_WINDOW)

    # -----------------------------------------------------
    # DURATION COUNTER
    # -----------------------------------------------------
    inactive_streak = 0

    # -----------------------------------------------------
    # OUTPUT LISTS
    # -----------------------------------------------------
    filtered_baby_values = []
    filtered_behavior_values = []
    filtered_audio_values = []

    smoothed_baby_values = []
    smoothed_behavior_values = []
    smoothed_audio_values = []
    smoothed_cry_probability_values = []

    inactive_streak_values = []
    recent_sudden_count_values = []

    risk_score_values = []
    risk_label_values = []
    risk_reason_values = []

    # -----------------------------------------------------
    # PROCESS EACH SAMPLE
    # -----------------------------------------------------
    for _, row in df.iterrows():

        # =================================================
        # YOLO CONFIDENCE FILTERING
        # =================================================
        baby_detected = safe_int(
            row["baby_detected"]
        )

        yolo_confidence = safe_float(
            row["yolo_confidence"]
        )

        if (
            baby_detected == 1
            and yolo_confidence >= YOLO_CONFIDENCE_THRESHOLD
        ):
            filtered_baby = 1
        else:
            filtered_baby = 0

        # =================================================
        # BEHAVIOR CONFIDENCE FILTERING
        # =================================================
        behavior = normalize_behavior(
            row["lstm_behavior"]
        )

        behavior_confidence = safe_float(
            row["behavior_confidence"]
        )

        pose_detected = safe_int(
            row["pose_detected"]
        )

        if (
            pose_detected == 1
            and behavior_confidence >= BEHAVIOR_CONFIDENCE_THRESHOLD
            and behavior != "Unknown"
        ):
            filtered_behavior = behavior
        else:
            filtered_behavior = "Unknown"

        # =================================================
        # AUDIO CONFIDENCE FILTERING
        # =================================================
        audio = normalize_audio(
            row["audio_class"]
        )

        audio_confidence = safe_float(
            row["audio_confidence"]
        )

        cry_probability = safe_float(
            row["cry_probability"]
        )

        if (
            audio_confidence >= AUDIO_CONFIDENCE_THRESHOLD
            and audio != "Unknown"
        ):
            filtered_audio = audio
        else:
            filtered_audio = "Unknown"

        filtered_baby_values.append(
            filtered_baby
        )

        filtered_behavior_values.append(
            filtered_behavior
        )

        filtered_audio_values.append(
            filtered_audio
        )

        # =================================================
        # ADD CURRENT VALUES TO HISTORY
        # =================================================
        baby_history.append(
            filtered_baby
        )

        behavior_history.append(
            filtered_behavior
        )

        audio_history.append(
            filtered_audio
        )

        cry_probability_history.append(
            cry_probability
        )

        # =================================================
        # BABY TEMPORAL SMOOTHING
        # =================================================
        if len(baby_history) >= MINIMUM_HISTORY:
            baby_positive_count = sum(baby_history)

            if baby_positive_count >= len(baby_history) / 2:
                smoothed_baby = 1
            else:
                smoothed_baby = 0
        else:
            smoothed_baby = filtered_baby

        # =================================================
        # BEHAVIOR TEMPORAL SMOOTHING
        # =================================================
        if len(behavior_history) >= MINIMUM_HISTORY:
            smoothed_behavior = majority_value(
                behavior_history,
                filtered_behavior,
            )
        else:
            smoothed_behavior = filtered_behavior

        # =================================================
        # AUDIO TEMPORAL SMOOTHING
        # =================================================
        if len(audio_history) >= MINIMUM_HISTORY:
            smoothed_audio = majority_value(
                audio_history,
                filtered_audio,
            )
        else:
            smoothed_audio = filtered_audio

        # =================================================
        # CRY PROBABILITY TEMPORAL SMOOTHING
        # =================================================
        if len(cry_probability_history) > 0:
            smoothed_cry_probability = (
                sum(cry_probability_history)
                / len(cry_probability_history)
            )
        else:
            smoothed_cry_probability = cry_probability

        # =================================================
        # TEMPORAL BEHAVIOR COUNTERS
        # =================================================
        if smoothed_baby == 0:
            inactive_streak = 0
            sudden_history.clear()

        else:
            if smoothed_behavior == "Inactive":
                inactive_streak += 1
            else:
                inactive_streak = 0

            if smoothed_behavior == "Sudden_Movement":
                sudden_history.append(1)
            else:
                sudden_history.append(0)

        recent_sudden_count = sum(sudden_history)

        # =================================================
        # FINAL RISK CALCULATION
        # =================================================
        (
            risk_score,
            risk_label,
            risk_reason,
        ) = calculate_risk(
            baby_detected=smoothed_baby,
            behavior=smoothed_behavior,
            audio=smoothed_audio,
            cry_probability=smoothed_cry_probability,
            inactive_streak=inactive_streak,
            recent_sudden_count=recent_sudden_count,
        )

        # =================================================
        # SAVE OUTPUT VALUES
        # =================================================
        smoothed_baby_values.append(
            smoothed_baby
        )

        smoothed_behavior_values.append(
            smoothed_behavior
        )

        smoothed_audio_values.append(
            smoothed_audio
        )

        smoothed_cry_probability_values.append(
            smoothed_cry_probability
        )

        inactive_streak_values.append(
            inactive_streak
        )

        recent_sudden_count_values.append(
            recent_sudden_count
        )

        risk_score_values.append(
            risk_score
        )

        risk_label_values.append(
            risk_label
        )

        risk_reason_values.append(
            risk_reason
        )

    # =========================================================
    # ADD NEW COLUMNS
    # =========================================================
    df["filtered_baby_detected"] = (
        filtered_baby_values
    )

    df["filtered_behavior"] = (
        filtered_behavior_values
    )

    df["filtered_audio"] = (
        filtered_audio_values
    )

    df["smoothed_baby_detected"] = (
        smoothed_baby_values
    )

    df["smoothed_behavior"] = (
        smoothed_behavior_values
    )

    df["smoothed_audio"] = (
        smoothed_audio_values
    )

    df["smoothed_cry_probability"] = (
        smoothed_cry_probability_values
    )

    df["inactive_streak"] = (
        inactive_streak_values
    )

    df["recent_sudden_count"] = (
        recent_sudden_count_values
    )

    df["risk_score"] = (
        risk_score_values
    )

    df["risk_label"] = (
        risk_label_values
    )

    df["risk_reason"] = (
        risk_reason_values
    )

    # =========================================================
    # REMOVE PERSON FEATURES
    # =========================================================
    person_columns = [
        "person_detected",
        "person_count",
    ]

    existing_person_columns = [
        column
        for column in person_columns
        if column in df.columns
    ]

    if existing_person_columns:
        df = df.drop(
            columns=existing_person_columns
        )

    # =========================================================
    # REMOVE PREVIOUS RULE-BASED STATUS
    # =========================================================
    leakage_columns = [
        "distress_detected",
        "distress_status",
    ]

    existing_leakage_columns = [
        column
        for column in leakage_columns
        if column in df.columns
    ]

    if existing_leakage_columns:
        df = df.drop(
            columns=existing_leakage_columns
        )

    # =========================================================
    # SAVE FINAL DATASET
    # =========================================================
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =========================================================
    # CREATE RISK SUMMARY
    # =========================================================
    class_order = [
        "No_Baby",
        "Low_Risk",
        "Medium_Risk",
        "High_Risk",
    ]

    summary = (
        df["risk_label"]
        .value_counts()
        .reindex(class_order, fill_value=0)
        .rename_axis("risk_label")
        .reset_index(name="sample_count")
    )

    summary["percentage"] = (
        summary["sample_count"]
        / len(df)
        * 100
    ).round(2)

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # =========================================================
    # TERMINAL OUTPUT
    # =========================================================
    print()
    print("=" * 70)
    print("RISK LABEL GENERATION COMPLETED")
    print("=" * 70)

    print(f"Final rows: {len(df)}")

    print(
        "Output dataset: "
        f"{os.path.abspath(OUTPUT_CSV)}"
    )

    print(
        "Risk summary: "
        f"{os.path.abspath(SUMMARY_CSV)}"
    )

    print()
    print("Risk Level Summary")
    print("-" * 70)
    print(summary.to_string(index=False))

    print()
    print("Final output classes:")
    print("1. No_Baby")
    print("2. Low_Risk")
    print("3. Medium_Risk")
    print("4. High_Risk")

    print()
    print("Old output CSV files were overwritten automatically.")
    print("You do not need to delete the old CSV files.")


if __name__ == "__main__":
    main()