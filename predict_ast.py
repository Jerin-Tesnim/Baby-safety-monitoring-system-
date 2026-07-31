import os
import sys
import json

import numpy as np
import librosa
import torch
import pandas as pd

from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ast_audio_model"
)

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 10
MAX_AUDIO_LENGTH = SAMPLE_RATE * MAX_AUDIO_SECONDS


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Check model
# ============================================================

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        "ast_audio_model folder পাওয়া যায়নি। "
        "আগে train_ast.py run করুন।"
    )


# ============================================================
# Load model and feature extractor
# ============================================================

print("Loading AST model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_DIR
)

model = ASTForAudioClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Get class mapping
# ============================================================

id2label = {
    int(key): value
    for key, value in model.config.id2label.items()
}

print("Classes:", id2label)


# ============================================================
# Load audio
# ============================================================

def load_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file পাওয়া যায়নি: {audio_path}"
        )

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        raise ValueError("Audio file empty.")

    # সর্বোচ্চ 10 seconds
    if len(audio) > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]

    return audio


# ============================================================
# Prediction
# ============================================================

def predict_audio(audio_path):
    audio = load_audio(audio_path)

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(
            input_values=input_values
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = id2label[predicted_id]
    confidence = float(
        probabilities[predicted_id].item()
    )

    probability_result = {}

    for class_id, class_name in id2label.items():
        probability_result[class_name] = float(
            probabilities[class_id].item()
        )

    return {
        "audio_file": audio_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_result
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "\nUsage:\n"
            'python predict_ast.py "test_audio.wav"'
        )
        sys.exit(1)

    test_audio_path = sys.argv[1]

    result = predict_audio(test_audio_path)

    print("\n" + "=" * 50)
    print("AST Audio Prediction")
    print("=" * 50)

    print("Audio:", result["audio_file"])
    print("Predicted Class:", result["predicted_class"])
    print(
        "Confidence:",
        f'{result["confidence"] * 100:.2f}%'
    )

    print("\nAll probabilities:")

    for class_name, probability in result["probabilities"].items():
        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

    output_row = {
        "audio_file": result["audio_file"],
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"]
    }

    for class_name, probability in result["probabilities"].items():
        output_row[f"{class_name}_probability"] = probability

    output_csv = os.path.join(
        BASE_DIR,
        "audio_prediction.csv"
    )

    pd.DataFrame([output_row]).to_csv(
        output_csv,
        index=False
    )

    print("\nPrediction saved:")
    print(output_csv)