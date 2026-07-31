import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    BASE_DIR,
    "live_ast_output.csv"
)


if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"CSV file was not found:\n{CSV_FILE}"
    )


df = pd.read_csv(CSV_FILE)

if "audio_class" not in df.columns:
    raise ValueError(
        "The CSV file does not contain the 'audio_class' column."
    )


counts = df["audio_class"].value_counts()


baby_cry_count = int(
    counts.get("Baby_Cry", 0)
)

noise_count = int(
    counts.get("Noise", 0)
)

normal_count = int(
    counts.get("Normal", 0)
)

total_count = len(df)


print("=" * 45)
print("AST LIVE AUDIO RESULT SUMMARY")
print("=" * 45)

print("Total Predictions :", total_count)
print("Baby Cry         :", baby_cry_count)
print("Noise            :", noise_count)
print("Normal           :", normal_count)

print("=" * 45)


summary_df = pd.DataFrame({
    "audio_class": [
        "Baby_Cry",
        "Noise",
        "Normal"
    ],
    "count": [
        baby_cry_count,
        noise_count,
        normal_count
    ]
})


SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "ast_result_summary.csv"
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\nSummary CSV saved:")
print(SUMMARY_FILE)