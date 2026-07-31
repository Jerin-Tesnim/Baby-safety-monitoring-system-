import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================
# SETTINGS
# =========================================================
MODEL_FILE = "best_tree_model_for_shap.pkl"
DATA_FILE = "final_fusion_risk_dataset.csv"
OUTPUT_DIR = "shap_results_combined"

SHAP_SAMPLE_SIZE = 700
TEMPORAL_WINDOW = 15
RANDOM_STATE = 42
LOCAL_SAMPLE_POSITION = 0


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_filename(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_").lower()


def save_figure(filename, width=12, height=8):
    figure = plt.gcf()
    figure.set_size_inches(width, height)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def get_processed_feature_names(preprocessor, fallback_size):
    try:
        names = list(preprocessor.get_feature_names_out())
        cleaned_names = []

        for name in names:
            name = str(name)

            if "__" in name:
                name = name.split("__", 1)[1]

            cleaned_names.append(name)

        return cleaned_names

    except Exception:
        return [
            f"feature_{index}"
            for index in range(fallback_size)
        ]


# =========================================================
# TEMPORAL FEATURE ENGINEERING
# Must match the training script
# =========================================================
def add_temporal_features(df):
    result = df.copy()

    temporal_numeric_columns = [
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

    available_columns = [
        column
        for column in temporal_numeric_columns
        if column in result.columns
    ]

    for column in available_columns:
        series = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        rolling = series.rolling(
            window=TEMPORAL_WINDOW,
            min_periods=1,
        )

        result[f"{column}_roll_mean"] = rolling.mean()
        result[f"{column}_roll_std"] = rolling.std().fillna(0.0)
        result[f"{column}_roll_min"] = rolling.min()
        result[f"{column}_roll_max"] = rolling.max()
        result[f"{column}_change"] = series.diff().fillna(0.0)

    if "audio_class" in result.columns:
        audio_text = (
            result["audio_class"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        for audio_class in ["cry", "noise", "normal"]:
            flag = (audio_text == audio_class).astype(int)

            result[f"audio_{audio_class}_recent_ratio"] = (
                flag.rolling(
                    window=TEMPORAL_WINDOW,
                    min_periods=1,
                ).mean()
            )

    if "lstm_behavior" in result.columns:
        behavior_text = (
            result["lstm_behavior"]
            .astype(str)
            .str.strip()
        )

        behavior_classes = [
            "Inactive",
            "Low_Movement",
            "Normal_Movement",
            "Restless_Movement",
            "Sudden_Movement",
        ]

        for behavior_class in behavior_classes:
            flag = (
                behavior_text == behavior_class
            ).astype(int)

            result[
                f"behavior_{behavior_class.lower()}_recent_ratio"
            ] = flag.rolling(
                window=TEMPORAL_WINDOW,
                min_periods=1,
            ).mean()

    return result


# =========================================================
# NORMALIZE XGBOOST NATIVE SHAP OUTPUT
# =========================================================
def normalize_contributions(
    contributions,
    number_of_classes,
    number_of_features,
):
    values = np.asarray(contributions)

    # Multiclass normal shape:
    # samples x classes x (features + bias)
    if (
        values.ndim == 3
        and values.shape[1] == number_of_classes
        and values.shape[2] == number_of_features + 1
    ):
        return values

    # Alternative:
    # samples x (features + bias) x classes
    if (
        values.ndim == 3
        and values.shape[2] == number_of_classes
        and values.shape[1] == number_of_features + 1
    ):
        return np.transpose(values, (0, 2, 1))

    # Binary/single-output fallback
    if (
        values.ndim == 2
        and values.shape[1] == number_of_features + 1
    ):
        return values[:, np.newaxis, :]

    raise ValueError(
        f"Unexpected native SHAP contribution shape: {values.shape}"
    )


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("COMBINED SHAP EXPLANATION")
    print("=" * 78)

    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file not found: {MODEL_FILE}")
        return

    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset file not found: {DATA_FILE}")
        return

    model_package = joblib.load(MODEL_FILE)

    pipeline = model_package["pipeline"]
    feature_columns = model_package["feature_columns"]
    class_names = list(model_package["class_names"])
    model_name = model_package.get("model_name", "XGBoost")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    print(f"Loaded model       : {model_name}")
    print(f"Risk classes       : {class_names}")

    df = pd.read_csv(DATA_FILE)

    if "sample_id" in df.columns:
        df["sample_id"] = pd.to_numeric(
            df["sample_id"],
            errors="coerce",
        )

        df = df.sort_values(
            by="sample_id"
        ).reset_index(drop=True)

    df = add_temporal_features(df)

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        print("Missing trained features:")

        for feature in missing_features:
            print(f"- {feature}")

        return

    X = df[feature_columns].copy()

    sample_size = min(SHAP_SAMPLE_SIZE, len(X))

    X_sample = X.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    processed_data = preprocessor.transform(X_sample)

    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    processed_data = np.asarray(
        processed_data,
        dtype=np.float32,
    )

    feature_names = get_processed_feature_names(
        preprocessor,
        processed_data.shape[1],
    )

    print(f"Rows used for SHAP : {sample_size}")
    print(f"Processed features : {len(feature_names)}")

    # -----------------------------------------------------
    # XGBOOST NATIVE SHAP CONTRIBUTIONS
    # This avoids the SHAP TreeExplainer base_score error.
    # -----------------------------------------------------
    booster = classifier.get_booster()
    dmatrix = xgb.DMatrix(processed_data)

    raw_contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        strict_shape=True,
    )

    contributions = normalize_contributions(
        raw_contributions,
        len(class_names),
        len(feature_names),
    )

    # Model predicted class for every sampled row
    predicted_probabilities = classifier.predict_proba(
        processed_data
    )

    predicted_class_indices = np.argmax(
        predicted_probabilities,
        axis=1,
    )

    predicted_class_names = [
        class_names[index]
        for index in predicted_class_indices
    ]

    # Select SHAP values for the class predicted for each row.
    # This creates one combined explanation instead of four separate sets.
    selected_shap_values = np.zeros(
        (
            sample_size,
            len(feature_names),
        ),
        dtype=float,
    )

    selected_base_values = np.zeros(
        sample_size,
        dtype=float,
    )

    for row_index, class_index in enumerate(
        predicted_class_indices
    ):
        selected_shap_values[row_index] = (
            contributions[
                row_index,
                class_index,
                :-1,
            ]
        )

        selected_base_values[row_index] = (
            contributions[
                row_index,
                class_index,
                -1,
            ]
        )

    # -----------------------------------------------------
    # SAVE COMBINED CSV
    # -----------------------------------------------------
    mean_absolute_shap = np.mean(
        np.abs(selected_shap_values),
        axis=0,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": mean_absolute_shap,
        }
    ).sort_values(
        by="mean_absolute_shap",
        ascending=False,
    )

    importance_df["rank"] = np.arange(
        1,
        len(importance_df) + 1,
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "combined_shap_feature_importance.csv",
        ),
        index=False,
    )

    prediction_df = pd.DataFrame(
        {
            "sample_position": np.arange(sample_size),
            "predicted_class": predicted_class_names,
            "prediction_confidence": np.max(
                predicted_probabilities,
                axis=1,
            ),
        }
    )

    prediction_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "shap_sample_predictions.csv",
        ),
        index=False,
    )

    # -----------------------------------------------------
    # FIGURE 1: COMBINED GLOBAL FEATURE IMPORTANCE BAR
    # -----------------------------------------------------
    top_bar = importance_df.head(20).sort_values(
        by="mean_absolute_shap",
        ascending=True,
    )

    plt.figure()

    plt.barh(
        top_bar["feature"],
        top_bar["mean_absolute_shap"],
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(
        "Combined Global SHAP Feature Importance"
    )

    save_figure(
        "01_combined_global_shap_bar.png",
        width=12,
        height=9,
    )

    # -----------------------------------------------------
    # FIGURE 2: ONE COMBINED BEESWARM
    # -----------------------------------------------------
    plt.figure()

    shap.summary_plot(
        selected_shap_values,
        processed_data,
        feature_names=feature_names,
        plot_type="dot",
        max_display=20,
        show=False,
    )

    plt.title(
        "Combined SHAP Beeswarm for Predicted Risk Classes"
    )

    save_figure(
        "02_combined_shap_beeswarm.png",
        width=13,
        height=9,
    )

    # -----------------------------------------------------
    # FIGURE 3: ONE WATERFALL FOR ONE PREDICTION
    # -----------------------------------------------------
    local_position = min(
        LOCAL_SAMPLE_POSITION,
        sample_size - 1,
    )

    local_class_index = predicted_class_indices[
        local_position
    ]

    local_class_name = class_names[
        local_class_index
    ]

    local_explanation = shap.Explanation(
        values=selected_shap_values[
            local_position
        ],
        base_values=selected_base_values[
            local_position
        ],
        data=processed_data[
            local_position
        ],
        feature_names=feature_names,
    )

    plt.figure()

    shap.plots.waterfall(
        local_explanation,
        max_display=20,
        show=False,
    )

    plt.title(
        f"Local SHAP Explanation - Predicted Class: {local_class_name}"
    )

    save_figure(
        "03_local_shap_waterfall.png",
        width=13,
        height=9,
    )

    # -----------------------------------------------------
    # FIGURE 4: ONE DEPENDENCE PLOT FOR TOP FEATURE
    # -----------------------------------------------------
    top_feature_name = importance_df.iloc[0][
        "feature"
    ]

    top_feature_index = feature_names.index(
        top_feature_name
    )

    plt.figure()

    shap.dependence_plot(
        top_feature_index,
        selected_shap_values,
        processed_data,
        feature_names=feature_names,
        interaction_index="auto",
        show=False,
    )

    plt.title(
        f"Combined SHAP Dependence Plot - {top_feature_name}"
    )

    save_figure(
        "04_top_feature_shap_dependence.png",
        width=11,
        height=8,
    )

    # -----------------------------------------------------
    # LOCAL SHAP CSV
    # -----------------------------------------------------
    local_df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": processed_data[
                local_position
            ],
            "shap_value": selected_shap_values[
                local_position
            ],
        }
    )

    local_df["absolute_shap_value"] = np.abs(
        local_df["shap_value"]
    )

    local_df = local_df.sort_values(
        by="absolute_shap_value",
        ascending=False,
    )

    local_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "local_shap_values.csv",
        ),
        index=False,
    )

    print()
    print("=" * 78)
    print("COMBINED SHAP EXPLANATION COMPLETED")
    print("=" * 78)
    print(f"Output folder : {os.path.abspath(OUTPUT_DIR)}")
    print("Generated figures:")
    print("1. 01_combined_global_shap_bar.png")
    print("2. 02_combined_shap_beeswarm.png")
    print("3. 03_local_shap_waterfall.png")
    print("4. 04_top_feature_shap_dependence.png")
    print()
    print("Generated CSV files:")
    print("1. combined_shap_feature_importance.csv")
    print("2. shap_sample_predictions.csv")
    print("3. local_shap_values.csv")


if __name__ == "__main__":
    main()