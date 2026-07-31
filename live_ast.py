import os
import time
import csv
import warnings

import numpy as np
import sounddevice as sd
import torch

from transformers import AutoFeatureExtractor, ASTForAudioClassification


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ast_audio_model"
)

CSV_FILE = os.path.join(
    BASE_DIR,
    "live_ast_output.csv"
)

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

# Internal Microphone device number
MIC_DEVICE = 1

# Very low sound will be classified as Normal
SILENCE_RMS_THRESHOLD = 0.009

warnings.filterwarnings("ignore")


# ============================================================
# Processing Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("LIVE AST AUDIO CLASSIFICATION")
print("=" * 60)

print("Processing Device:", device)


# ============================================================
# Check Model Folder
# ============================================================

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"Model folder was not found:\n{MODEL_DIR}"
    )


# ============================================================
# Load AST Model
# ============================================================

print("\nLoading AST model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_DIR
)

model = ASTForAudioClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

id2label = {
    int(label_id): label_name
    for label_id, label_name in model.config.id2label.items()
}

print("Model loaded successfully.")
print("Classes:", id2label)


# ============================================================
# Check Microphone
# ============================================================

try:
    mic_info = sd.query_devices(
        MIC_DEVICE,
        "input"
    )

    print("\nSelected Microphone:")
    print(mic_info["name"])

except Exception as error:
    print("\nAvailable Audio Devices:")
    print(sd.query_devices())

    raise RuntimeError(
        f"Microphone error: {error}"
    )


# ============================================================
# Create CSV File
# ============================================================

def create_csv_file():
    if not os.path.exists(CSV_FILE):
        with open(
            CSV_FILE,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "sample_id",
                "audio_class",
                "confidence",
                "sound_level",
                "decision_method",
                "baby_cry_probability",
                "noise_probability",
                "normal_probability"
            ])

        print("\nCSV file created:")
        print(CSV_FILE)

    else:
        print("\nExisting CSV file will be updated:")
        print(CSV_FILE)


create_csv_file()


# ============================================================
# Record Live Audio
# ============================================================

def record_audio():
    total_samples = int(
        RECORD_SECONDS * SAMPLE_RATE
    )

    recording = sd.rec(
        frames=total_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE
    )

    sd.wait()

    return recording.flatten()


# ============================================================
# AST Prediction
# ============================================================

def predict_ast(audio):
    if audio is None or len(audio) == 0:
        raise ValueError(
            "No audio data was received from the microphone."
        )

    rms = float(
        np.sqrt(np.mean(np.square(audio)))
    )

    # Very low sound means Normal
    if rms < SILENCE_RMS_THRESHOLD:
        return {
            "class": "Normal",
            "confidence": 1.0,
            "rms": rms,
            "probabilities": {
                "Baby_Cry": 0.0,
                "Noise": 0.0,
                "Normal": 1.0
            },
            "method": "RMS Threshold"
        }

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        logits = model(
            input_values=input_values
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = id2label[predicted_id]

    confidence = float(
        probabilities[predicted_id].item()
    )

    probability_dictionary = {}

    for class_id, class_name in id2label.items():
        probability_dictionary[class_name] = float(
            probabilities[class_id].item()
        )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "rms": rms,
        "probabilities": probability_dictionary,
        "method": "AST Model"
    }


# ============================================================
# Get Probability Safely
# ============================================================

def get_probability(probabilities, class_name):
    for label, value in probabilities.items():
        if label.lower() == class_name.lower():
            return float(value)

    return 0.0


# ============================================================
# Save Prediction to CSV
# ============================================================

def save_to_csv(sample_id, result):
    baby_cry_probability = get_probability(
        result["probabilities"],
        "Baby_Cry"
    )

    noise_probability = get_probability(
        result["probabilities"],
        "Noise"
    )

    normal_probability = get_probability(
        result["probabilities"],
        "Normal"
    )

    with open(
        CSV_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            sample_id,
            result["class"],
            round(result["confidence"], 6),
            round(result["rms"], 6),
            result["method"],
            round(baby_cry_probability, 6),
            round(noise_probability, 6),
            round(normal_probability, 6)
        ])


# ============================================================
# Display Result
# ============================================================

def show_result(result):
    print("\n" + "-" * 60)

    print("Audio Class    :", result["class"])

    print(
        "Confidence     :",
        f'{result["confidence"] * 100:.2f}%'
    )

    print(
        "Sound Level    :",
        f'{result["rms"]:.6f}'
    )

    print(
        "Decision Method:",
        result["method"]
    )

    print("\nClass Probabilities:")

    for class_name, probability in result["probabilities"].items():
        print(
            f"  {class_name:<12}: "
            f"{probability * 100:.2f}%"
        )

    print("-" * 60)


# ============================================================
# Live Prediction Loop
# ============================================================

print("\nLive microphone prediction started.")

print(
    f"Audio will be recorded for "
    f"{RECORD_SECONDS} seconds per prediction."
)

print("Press Ctrl + C to stop live prediction.")

prediction_number = 1

try:
    while True:
        print(
            f"\nPrediction {prediction_number}: "
            f"Provide sound for {RECORD_SECONDS} seconds..."
        )

        audio_data = record_audio()

        print("Audio recorded. Processing with AST model...")

        result = predict_ast(
            audio_data
        )

        show_result(
            result
        )

        save_to_csv(
            prediction_number,
            result
        )

        print("Prediction saved to CSV.")

        prediction_number += 1

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n\nLive prediction stopped.")

    print("CSV output saved at:")
    print(CSV_FILE)

except Exception as error:
    print("\nLive prediction error:")
    print(error)