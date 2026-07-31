import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "audio_dataset")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "ast_audio_model")
RESULT_DIR = os.path.join(BASE_DIR, "ast_results")

PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 10
MAX_AUDIO_LENGTH = SAMPLE_RATE * MAX_AUDIO_SECONDS

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

SUPPORTED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac"
)

warnings.filterwarnings("ignore")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("AST Audio Classification Training")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU পাওয়া যায়নি। Training CPU-তে চলবে।")


# ============================================================
# Create output folders
# ============================================================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Read dataset
# ============================================================

def collect_audio_files(dataset_directory):
    if not os.path.exists(dataset_directory):
        raise FileNotFoundError(
            f"Dataset folder পাওয়া যায়নি:\n{dataset_directory}"
        )

    class_names = sorted(
        folder_name
        for folder_name in os.listdir(dataset_directory)
        if os.path.isdir(os.path.join(dataset_directory, folder_name))
    )

    if len(class_names) < 2:
        raise ValueError(
            "Dataset folder-এর ভিতরে অন্তত দুইটি class folder থাকতে হবে।"
        )

    label2id = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    id2label = {
        index: class_name
        for class_name, index in label2id.items()
    }

    records = []

    for class_name in class_names:
        class_folder = os.path.join(dataset_directory, class_name)

        for root, _, files in os.walk(class_folder):
            for file_name in files:
                if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                    file_path = os.path.join(root, file_name)

                    records.append({
                        "file_path": file_path,
                        "class_name": class_name,
                        "label": label2id[class_name]
                    })

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise ValueError(
            "Dataset folder-এ কোনো supported audio file পাওয়া যায়নি।"
        )

    return dataframe, class_names, label2id, id2label


df, class_names, label2id, id2label = collect_audio_files(DATASET_DIR)

print("\nClasses:", class_names)
print("Total audio files:", len(df))

print("\nAudio files per class:")
print(df["class_name"].value_counts().sort_index())


# ============================================================
# Check minimum class size
# ============================================================

class_counts = df["class_name"].value_counts()

if class_counts.min() < 5:
    raise ValueError(
        "প্রতিটি class-এ কমপক্ষে 5টি audio file থাকতে হবে। "
        "ভালো result-এর জন্য 100 বা তার বেশি রাখা উচিত।"
    )


# ============================================================
# Train-validation split
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.to_csv(
    os.path.join(RESULT_DIR, "train_files.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(RESULT_DIR, "validation_files.csv"),
    index=False
)

print("\nTraining files:", len(train_df))
print("Validation files:", len(val_df))


# ============================================================
# Feature extractor
# ============================================================

print("\nLoading AST feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL
)


# ============================================================
# Audio augmentation
# ============================================================

def add_audio_augmentation(audio):
    """
    Training audio-তে ছোটখাটো পরিবর্তন করে নতুন variation তৈরি করে।
    """

    augmented_audio = audio.copy()

    # Random volume change
    if random.random() < 0.40:
        gain = random.uniform(0.75, 1.25)
        augmented_audio = augmented_audio * gain

    # Small Gaussian noise
    if random.random() < 0.30:
        noise_strength = random.uniform(0.001, 0.005)
        random_noise = np.random.randn(len(augmented_audio))
        augmented_audio = (
            augmented_audio + noise_strength * random_noise
        )

    # Small time shift
    if random.random() < 0.30:
        maximum_shift = int(0.10 * len(augmented_audio))

        if maximum_shift > 0:
            shift_amount = random.randint(
                -maximum_shift,
                maximum_shift
            )

            augmented_audio = np.roll(
                augmented_audio,
                shift_amount
            )

    augmented_audio = np.clip(
        augmented_audio,
        -1.0,
        1.0
    )

    return augmented_audio.astype(np.float32)


# ============================================================
# Audio loading
# ============================================================

def load_audio(file_path):
    try:
        audio, _ = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        audio = audio.astype(np.float32)

        if len(audio) == 0:
            raise ValueError("Audio file empty.")

        # খুব লম্বা audio হলে random 10-second অংশ নেওয়া হবে
        if len(audio) > MAX_AUDIO_LENGTH:
            maximum_start = len(audio) - MAX_AUDIO_LENGTH
            start_position = random.randint(0, maximum_start)

            audio = audio[
                start_position:
                start_position + MAX_AUDIO_LENGTH
            ]

        return audio

    except Exception as error:
        raise RuntimeError(
            f"Audio load করা যায়নি: {file_path}\nError: {error}"
        )


# ============================================================
# Custom Dataset
# ============================================================

class ASTAudioDataset(Dataset):

    def __init__(
        self,
        dataframe,
        extractor,
        training=False
    ):
        self.dataframe = dataframe
        self.extractor = extractor
        self.training = training

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        file_path = row["file_path"]
        label = int(row["label"])

        audio = load_audio(file_path)

        if self.training:
            audio = add_audio_augmentation(audio)

        extracted_features = self.extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = extracted_features["input_values"].squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long
            ),
            "file_path": file_path
        }


train_dataset = ASTAudioDataset(
    dataframe=train_df,
    extractor=feature_extractor,
    training=True
)

val_dataset = ASTAudioDataset(
    dataframe=val_df,
    extractor=feature_extractor,
    training=False
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Load AST model
# ============================================================

print("\nLoading pretrained AST model...")

model = ASTForAudioClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(class_names),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

model.to(device)

print("Model loaded successfully.")


# ============================================================
# Class weights
# ============================================================

training_class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
    .values
)

number_of_training_samples = len(train_df)
number_of_classes = len(class_names)

class_weights = (
    number_of_training_samples
    / (number_of_classes * training_class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("\nClass weights:", class_weights)


# ============================================================
# Optimizer and loss
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(current_model, data_loader):
    current_model.train()

    total_loss = 0.0
    predictions = []
    actual_labels = []

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        leave=False
    )

    for batch in progress_bar:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = current_model(
            input_values=input_values
        )

        logits = outputs.logits

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            current_model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        predicted_classes = torch.argmax(
            logits,
            dim=1
        )

        predictions.extend(
            predicted_classes.detach().cpu().numpy()
        )

        actual_labels.extend(
            labels.detach().cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / max(len(data_loader), 1)

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    return average_loss, accuracy


# ============================================================
# Validation function
# ============================================================

def validate_model(current_model, data_loader):
    current_model.eval()

    total_loss = 0.0
    predictions = []
    actual_labels = []
    probabilities = []
    file_paths = []

    with torch.no_grad():
        progress_bar = tqdm(
            data_loader,
            desc="Validation",
            leave=False
        )

        for batch in progress_bar:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = current_model(
                input_values=input_values
            )

            logits = outputs.logits

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            batch_probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_classes = torch.argmax(
                batch_probabilities,
                dim=1
            )

            predictions.extend(
                predicted_classes.cpu().numpy()
            )

            actual_labels.extend(
                labels.cpu().numpy()
            )

            probabilities.extend(
                batch_probabilities.cpu().numpy()
            )

            file_paths.extend(
                batch["file_path"]
            )

    average_loss = total_loss / max(len(data_loader), 1)

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    return (
        average_loss,
        accuracy,
        np.array(actual_labels),
        np.array(predictions),
        np.array(probabilities),
        file_paths
    )


# ============================================================
# Main training loop
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": [],
    "learning_rate": []
}

best_validation_accuracy = 0.0
best_validation_loss = float("inf")

print("\n" + "=" * 60)
print("Training started")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    (
        validation_loss,
        validation_accuracy,
        true_labels,
        predicted_labels,
        predicted_probabilities,
        validation_file_paths
    ) = validate_model(
        model,
        val_loader
    )

    scheduler.step(validation_loss)

    current_learning_rate = optimizer.param_groups[0]["lr"]

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)
    history["learning_rate"].append(current_learning_rate)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_learning_rate:.8f}"
    )

    # Save best model
    if (
        validation_accuracy > best_validation_accuracy
        or (
            validation_accuracy == best_validation_accuracy
            and validation_loss < best_validation_loss
        )
    ):
        best_validation_accuracy = validation_accuracy
        best_validation_loss = validation_loss

        model.save_pretrained(MODEL_OUTPUT_DIR)
        feature_extractor.save_pretrained(MODEL_OUTPUT_DIR)

        with open(
            os.path.join(MODEL_OUTPUT_DIR, "labels.json"),
            "w",
            encoding="utf-8"
        ) as label_file:
            json.dump(
                {
                    "class_names": class_names,
                    "label2id": label2id,
                    "id2label": {
                        str(key): value
                        for key, value in id2label.items()
                    },
                    "sample_rate": SAMPLE_RATE,
                    "maximum_audio_seconds": MAX_AUDIO_SECONDS
                },
                label_file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "Best model saved:",
            MODEL_OUTPUT_DIR
        )


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(history)

history_csv_path = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_csv_path,
    index=False
)


# ============================================================
# Load best model for final evaluation
# ============================================================

print("\nLoading best model for final evaluation...")

best_model = ASTForAudioClassification.from_pretrained(
    MODEL_OUTPUT_DIR
)

best_model.to(device)

(
    final_validation_loss,
    final_validation_accuracy,
    true_labels,
    predicted_labels,
    predicted_probabilities,
    validation_file_paths
) = validate_model(
    best_model,
    val_loader
)


# ============================================================
# Classification report
# ============================================================

report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 60)
print("Final Classification Report")
print("=" * 60)
print(report_text)

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as report_file:
    report_file.write(report_text)


report_dictionary = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_dataframe = pd.DataFrame(
    report_dictionary
).transpose()

report_dataframe.to_csv(
    os.path.join(
        RESULT_DIR,
        "classification_report.csv"
    )
)


# ============================================================
# Save validation predictions
# ============================================================

prediction_records = []

for index, file_path in enumerate(validation_file_paths):
    true_id = int(true_labels[index])
    predicted_id = int(predicted_labels[index])

    record = {
        "file_path": file_path,
        "actual_class": id2label[true_id],
        "predicted_class": id2label[predicted_id],
        "prediction_confidence": float(
            predicted_probabilities[index][predicted_id]
        )
    }

    for class_id, class_name in id2label.items():
        record[f"{class_name}_probability"] = float(
            predicted_probabilities[index][class_id]
        )

    prediction_records.append(record)


predictions_df = pd.DataFrame(
    prediction_records
)

predictions_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "validation_predictions.csv"
    ),
    index=False
)


# ============================================================
# Confusion matrix
# ============================================================

confusion_matrix_values = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_values,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=plt.gca()
)

plt.title("AST Audio Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Accuracy graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["validation_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("AST Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Loss graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AST Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 60)
print("Training completed successfully")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)

print("Saved model folder:")
print(MODEL_OUTPUT_DIR)

print("\nSaved result folder:")
print(RESULT_DIR)

print("\nGenerated files:")
print("1. ast_audio_model/")
print("2. training_history.csv")
print("3. classification_report.txt")
print("4. classification_report.csv")
print("5. validation_predictions.csv")
print("6. confusion_matrix.png")
print("7. accuracy_curve.png")
print("8. loss_curve.png")