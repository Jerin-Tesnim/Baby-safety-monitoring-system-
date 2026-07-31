import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# =========================================================
# FILE PATHS
# Same old file names will be overwritten automatically
# =========================================================
INPUT_CSV = "final_fusion_risk_dataset.csv"
RESULTS_CSV = "model_comparison_results.csv"
PREDICTIONS_CSV = "best_model_test_predictions.csv"
BEST_MODEL_FILE = "best_accuracy_risk_model.pkl"
BEST_TREE_MODEL_FILE = "best_tree_model_for_shap.pkl"
LABEL_ENCODER_FILE = "risk_label_encoder.pkl"
FEATURE_LIST_FILE = "risk_model_features.txt"

OUTPUT_DIR = "model_results"
CONFUSION_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")


# =========================================================
# SETTINGS
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
RUN_SVM = True
PRIMARY_METRIC = "Accuracy"

# Number of recent samples used for temporal features
TEMPORAL_WINDOW = 15


# =========================================================
# OPTIONAL MODEL IMPORTS
# =========================================================
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# =========================================================
# PREPROCESSING
# =========================================================
def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def create_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


# =========================================================
# TEMPORAL FEATURE ENGINEERING
# =========================================================
def add_temporal_features(df):
    """
    Create additional features only from original raw multimodal outputs.

    Leakage columns such as risk_score, risk_reason, filtered_*,
    smoothed_*, inactive_streak and recent_sudden_count are not used.
    """

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

    available_numeric_columns = [
        column
        for column in temporal_numeric_columns
        if column in result.columns
    ]

    for column in available_numeric_columns:
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

            safe_name = behavior_class.lower()

            result[f"behavior_{safe_name}_recent_ratio"] = (
                flag.rolling(
                    window=TEMPORAL_WINDOW,
                    min_periods=1,
                ).mean()
            )

    return result


# =========================================================
# CHART FUNCTIONS
# =========================================================
def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    model_name,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=30,
    )

    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    path = os.path.join(
        CONFUSION_DIR,
        filename,
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_comparison_charts(results_df):
    """
    Saves:
    1. Old accuracy-only chart
    2. Accuracy, Precision, Recall and F1-score grouped chart
    """

    # -----------------------------------------------------
    # OLD ACCURACY-ONLY CHART
    # -----------------------------------------------------
    accuracy_df = results_df.sort_values(
        by="Accuracy",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(
        accuracy_df["Model"],
        accuracy_df["Accuracy"],
    )

    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Model")
    ax.set_title("Risk Classification Model Comparison")
    ax.set_xlim(0, 1.05)

    for index, value in enumerate(
        accuracy_df["Accuracy"]
    ):
        ax.text(
            value + 0.005,
            index,
            f"{value:.4f}",
            va="center",
        )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_accuracy_comparison.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------
    # ALL METRICS GROUPED BAR CHART
    # -----------------------------------------------------
    chart_df = results_df.sort_values(
        by="Accuracy",
        ascending=False,
    ).reset_index(drop=True)

    model_names = chart_df["Model"].tolist()

    metric_columns = [
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
    ]

    x_positions = np.arange(
        len(model_names)
    )

    bar_width = 0.19

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    for metric_index, metric_name in enumerate(
        metric_columns
    ):
        offset = (
            metric_index
            - (len(metric_columns) - 1) / 2
        ) * bar_width

        bars = ax.bar(
            x_positions + offset,
            chart_df[metric_name],
            bar_width,
            label=metric_name.replace("_", " "),
        )

        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.006,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
            )

    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Score")
    ax.set_title(
        "Model Comparison: Accuracy, Precision, Recall and F1-score"
    )

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        model_names,
        rotation=30,
        ha="right",
    )

    ax.set_ylim(0.0, 1.08)
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "all_metrics_comparison.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MODEL DEFINITIONS
# =========================================================
def build_models(number_of_classes):
    models = {}

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            objective="multiclass",
            num_class=number_of_classes,
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=15,
            subsample=0.95,
            colsample_bytree=0.95,
            reg_alpha=0.02,
            reg_lambda=0.30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=number_of_classes,
            n_estimators=650,
            max_depth=8,
            learning_rate=0.035,
            min_child_weight=1,
            subsample=0.95,
            colsample_bytree=0.95,
            gamma=0.0,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=800,
            depth=9,
            learning_rate=0.04,
            l2_leaf_reg=4.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.95,
        random_state=RANDOM_STATE,
    )

    models["Decision Tree"] = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )

    models["Logistic Regression"] = LogisticRegression(
        C=5.0,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    if RUN_SVM:
        models["SVM"] = SVC(
            kernel="rbf",
            C=20.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
            cache_size=2000,
        )

    return models


# =========================================================
# MAIN PROGRAM
# =========================================================
def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        CONFUSION_DIR,
        exist_ok=True,
    )

    print("=" * 82)
    print("MULTIMODAL RISK MODEL COMPARISON")
    print("=" * 82)

    if not os.path.exists(INPUT_CSV):
        print(
            f"Error: File not found: {INPUT_CSV}"
        )
        return

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"Dataset rows   : {len(df)}"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    if "risk_label" not in df.columns:
        print(
            "Error: 'risk_label' column was not found."
        )
        return

    # Sort samples before making temporal features
    if "sample_id" in df.columns:
        df["sample_id"] = pd.to_numeric(
            df["sample_id"],
            errors="coerce",
        )

        df = df.sort_values(
            by="sample_id",
        ).reset_index(drop=True)

    df = add_temporal_features(
        df
    )

    # -----------------------------------------------------
    # NEVER USE THESE COLUMNS AS MODEL INPUT
    # -----------------------------------------------------
    forbidden_columns = {
        "sample_id",
        "risk_label",
        "risk_score",
        "risk_reason",
        "filtered_baby_detected",
        "filtered_behavior",
        "filtered_audio",
        "smoothed_baby_detected",
        "smoothed_behavior",
        "smoothed_audio",
        "smoothed_cry_probability",
        "inactive_streak",
        "low_movement_streak",
        "restless_streak",
        "recent_sudden_count",
        "distress_detected",
        "distress_status",
        "person_detected",
        "person_count",
    }

    requested_categorical_features = [
        "lstm_behavior",
        "audio_class",
    ]

    categorical_features = [
        column
        for column in requested_categorical_features
        if column in df.columns
    ]

    numeric_features = []

    for column in df.columns:
        if column in forbidden_columns:
            continue

        if column in categorical_features:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if converted.notna().sum() > 0:
            df[column] = converted
            numeric_features.append(column)

    feature_columns = (
        numeric_features
        + categorical_features
    )

    if not feature_columns:
        print(
            "Error: No valid training features were found."
        )
        return

    print()
    print(
        f"Total model features: {len(feature_columns)}"
    )

    print()
    print("Features used:")

    for feature in feature_columns:
        print(f"- {feature}")

    df = df.dropna(
        subset=["risk_label"]
    ).copy()

    X = df[
        feature_columns
    ].copy()

    y_text = (
        df["risk_label"]
        .astype(str)
        .str.strip()
    )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_text
    )

    class_names = list(
        label_encoder.classes_
    )

    number_of_classes = len(
        class_names
    )

    print()
    print(
        f"Classes ({number_of_classes}): {class_names}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows : {len(X_test)}"
    )

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    models = build_models(
        number_of_classes
    )

    expected_models = [
        "XGBoost",
        "Random Forest",
        "SVM",
        "Logistic Regression",
        "LightGBM",
        "KNN",
        "Gradient Boosting",
        "Decision Tree",
        "CatBoost",
    ]

    unavailable_models = [
        name
        for name in expected_models
        if name not in models
    ]

    if unavailable_models:
        print()
        print(
            "Models skipped because packages/settings are unavailable:"
        )

        for name in unavailable_models:
            print(f"- {name}")

    results = []
    trained_pipelines = {}
    prediction_store = {}

    print()
    print("=" * 82)
    print("MODEL TRAINING STARTED")
    print("=" * 82)

    for model_name, classifier in models.items():
        print()
        print("-" * 82)
        print(f"Training: {model_name}")
        print("-" * 82)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        start_time = time.time()

        try:
            pipeline.fit(
                X_train,
                y_train,
            )

            y_pred = pipeline.predict(
                X_test
            )

            elapsed_seconds = (
                time.time()
                - start_time
            )

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            precision_macro = precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            recall_macro = recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            f1_macro = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            f1_weighted = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            report = classification_report(
                y_test,
                y_pred,
                labels=np.arange(
                    number_of_classes
                ),
                target_names=class_names,
                zero_division=0,
            )

            print(
                f"Accuracy          : {accuracy:.4f}"
            )

            print(
                f"Macro Precision   : {precision_macro:.4f}"
            )

            print(
                f"Macro Recall      : {recall_macro:.4f}"
            )

            print(
                f"Macro F1-score    : {f1_macro:.4f}"
            )

            print(
                f"Weighted F1-score : {f1_weighted:.4f}"
            )

            print(
                f"Training time     : {elapsed_seconds:.2f} seconds"
            )

            print()
            print(report)

            report_filename = (
                model_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                + "_classification_report.txt"
            )

            with open(
                os.path.join(
                    OUTPUT_DIR,
                    report_filename,
                ),
                "w",
                encoding="utf-8",
            ) as report_file:
                report_file.write(
                    report
                )

            save_confusion_matrix(
                y_test,
                y_pred,
                class_names,
                model_name,
            )

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Macro_Precision": precision_macro,
                    "Macro_Recall": recall_macro,
                    "Macro_F1": f1_macro,
                    "Weighted_F1": f1_weighted,
                    "Training_Time_Seconds": elapsed_seconds,
                }
            )

            trained_pipelines[
                model_name
            ] = pipeline

            prediction_store[
                model_name
            ] = y_pred

        except Exception as error:
            print(
                f"Failed: {model_name}"
            )

            print(
                f"Reason: {error}"
            )

    if not results:
        print(
            "Error: No model trained successfully."
        )
        return

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            PRIMARY_METRIC,
            "Macro_F1",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    # Overwrites old comparison CSV
    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # Overwrites old charts and creates the all-metrics chart
    save_comparison_charts(
        results_df
    )

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_pipeline = (
        trained_pipelines[
            best_model_name
        ]
    )

    best_predictions = (
        prediction_store[
            best_model_name
        ]
    )

    model_package = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "class_names": class_names,
        "test_accuracy": float(
            results_df.iloc[0]["Accuracy"]
        ),
        "macro_precision": float(
            results_df.iloc[0]["Macro_Precision"]
        ),
        "macro_recall": float(
            results_df.iloc[0]["Macro_Recall"]
        ),
        "macro_f1": float(
            results_df.iloc[0]["Macro_F1"]
        ),
        "temporal_window": TEMPORAL_WINDOW,
    }

    # Overwrites old best model
    joblib.dump(
        model_package,
        BEST_MODEL_FILE,
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_FILE,
    )

    with open(
        FEATURE_LIST_FILE,
        "w",
        encoding="utf-8",
    ) as feature_file:
        for feature in feature_columns:
            feature_file.write(
                feature + "\n"
            )

    prediction_df = (
        X_test
        .reset_index(drop=True)
        .copy()
    )

    prediction_df[
        "actual_risk_label"
    ] = label_encoder.inverse_transform(
        y_test
    )

    prediction_df[
        "predicted_risk_label"
    ] = label_encoder.inverse_transform(
        best_predictions.astype(int)
    )

    prediction_df[
        "prediction_correct"
    ] = (
        prediction_df[
            "actual_risk_label"
        ]
        == prediction_df[
            "predicted_risk_label"
        ]
    ).astype(int)

    try:
        probabilities = (
            best_pipeline.predict_proba(
                X_test
            )
        )

        for class_index, class_name in enumerate(
            class_names
        ):
            prediction_df[
                f"probability_{class_name}"
            ] = probabilities[
                :,
                class_index,
            ]

    except Exception:
        pass

    # Overwrites old predictions CSV
    prediction_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    tree_model_names = {
        "XGBoost",
        "Random Forest",
        "LightGBM",
        "Gradient Boosting",
        "Decision Tree",
        "CatBoost",
    }

    tree_results_df = results_df[
        results_df["Model"].isin(
            tree_model_names
        )
    ].copy()

    best_tree_name = None

    if not tree_results_df.empty:
        best_tree_name = (
            tree_results_df.iloc[0][
                "Model"
            ]
        )

        best_tree_pipeline = (
            trained_pipelines[
                best_tree_name
            ]
        )

        best_tree_package = {
            "model_name": best_tree_name,
            "pipeline": best_tree_pipeline,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "class_names": class_names,
            "test_accuracy": float(
                tree_results_df.iloc[0][
                    "Accuracy"
                ]
            ),
            "macro_precision": float(
                tree_results_df.iloc[0][
                    "Macro_Precision"
                ]
            ),
            "macro_recall": float(
                tree_results_df.iloc[0][
                    "Macro_Recall"
                ]
            ),
            "macro_f1": float(
                tree_results_df.iloc[0][
                    "Macro_F1"
                ]
            ),
            "temporal_window": TEMPORAL_WINDOW,
        }

        # Overwrites old SHAP model
        joblib.dump(
            best_tree_package,
            BEST_TREE_MODEL_FILE,
        )

    print()
    print("=" * 82)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 82)

    print()
    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Best accuracy model : {best_model_name}"
    )

    print(
        f"Best test accuracy  : "
        f"{results_df.iloc[0]['Accuracy']:.4f}"
    )

    print(
        f"Best macro precision: "
        f"{results_df.iloc[0]['Macro_Precision']:.4f}"
    )

    print(
        f"Best macro recall   : "
        f"{results_df.iloc[0]['Macro_Recall']:.4f}"
    )

    print(
        f"Best macro F1-score : "
        f"{results_df.iloc[0]['Macro_F1']:.4f}"
    )

    print(
        f"Best model saved    : "
        f"{os.path.abspath(BEST_MODEL_FILE)}"
    )

    if best_tree_name is not None:
        tree_row = (
            tree_results_df.iloc[0]
        )

        print()
        print(
            f"Best SHAP tree model: {best_tree_name}"
        )

        print(
            f"Tree test accuracy  : "
            f"{tree_row['Accuracy']:.4f}"
        )

        print(
            f"SHAP model saved    : "
            f"{os.path.abspath(BEST_TREE_MODEL_FILE)}"
        )

    print()
    print(
        f"Comparison CSV      : "
        f"{os.path.abspath(RESULTS_CSV)}"
    )

    print(
        f"Predictions CSV     : "
        f"{os.path.abspath(PREDICTIONS_CSV)}"
    )

    print(
        "Accuracy chart      : "
        f"{os.path.abspath(os.path.join(OUTPUT_DIR, 'model_accuracy_comparison.png'))}"
    )

    print(
        "All metrics chart   : "
        f"{os.path.abspath(os.path.join(OUTPUT_DIR, 'all_metrics_comparison.png'))}"
    )

    print(
        f"Confusion matrices  : "
        f"{os.path.abspath(CONFUSION_DIR)}"
    )

    print()
    print(
        "Old result files were overwritten automatically."
    )

    print(
        "Accuracy is not manually changed or forced."
    )

    print(
        "The labels remain pseudo-labels until they are reviewed "
        "and corrected by a human expert."
    )


if __name__ == "__main__":
    main()