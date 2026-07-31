import os
import cv2
import mediapipe as mp
import csv
import math
from copy import deepcopy


# ==========================================
# Settings
# ==========================================
CAMERA_INDEX = 0
OUTPUT_CSV = "mediapipe_33_landmarks_output.csv"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ==========================================
# MediaPipe Initialization
# ==========================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# Movement Calculation
# ==========================================
def calculate_movement(current_landmarks, previous_landmarks):
    """
    Calculates average movement using all 33 landmarks.
    """

    if current_landmarks is None or previous_landmarks is None:
        return 0.0

    total_movement = 0.0
    valid_landmark_count = 0

    for current, previous in zip(current_landmarks, previous_landmarks):
        dx = current.x - previous.x
        dy = current.y - previous.y
        dz = current.z - previous.z

        distance = math.sqrt(
            (dx * dx) +
            (dy * dy) +
            (dz * dz)
        )

        total_movement += distance
        valid_landmark_count += 1

    if valid_landmark_count == 0:
        return 0.0

    return total_movement / valid_landmark_count


# ==========================================
# Create CSV Header
# ==========================================
def create_csv_header():
    """
    Creates columns for:
    sample_id,
    pose_detected,
    movement_score,
    and x, y, z, visibility for all 33 landmarks.
    """

    header = [
        "sample_id",
        "pose_detected",
        "movement_score"
    ]

    for landmark_id in range(33):
        landmark_name = mp_pose.PoseLandmark(landmark_id).name.lower()

        header.extend([
            f"{landmark_id}_{landmark_name}_x",
            f"{landmark_id}_{landmark_name}_y",
            f"{landmark_id}_{landmark_name}_z",
            f"{landmark_id}_{landmark_name}_visibility"
        ])

    return header


# ==========================================
# Create CSV Row
# ==========================================
def create_csv_row(
    sample_id,
    pose_detected,
    movement_score,
    landmarks
):
    row = [
        sample_id,
        pose_detected,
        movement_score
    ]

    if pose_detected == 1 and landmarks is not None:
        for landmark in landmarks:
            row.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
    else:
        # 33 landmarks × 4 values
        row.extend([0.0] * (33 * 4))

    return row


# ==========================================
# Main Program
# ==========================================
def main():
    previous_landmarks = None
    sample_id = 1

    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        print("Error: Camera open করা যায়নি।")
        print("CAMERA_INDEX পরিবর্তন করে 1 অথবা 2 দিয়ে চেষ্টা করুন।")
        return

    csv_header = create_csv_header()

    with open(
        OUTPUT_CSV,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

        print("============================================")
        print("MediaPipe 33 Landmarks Data Collection")
        print("============================================")
        print(f"CSV file: {os.path.abspath(OUTPUT_CSV)}")
        print("ESC চাপলে program বন্ধ হবে।")
        print("============================================")

        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Camera থেকে frame পাওয়া যায়নি।")
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Improve processing
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            pose_detected = 0
            movement_score = 0.0
            current_landmarks = None

            if results.pose_landmarks:
                pose_detected = 1
                current_landmarks = results.pose_landmarks.landmark

                if previous_landmarks is not None:
                    movement_score = calculate_movement(
                        current_landmarks,
                        previous_landmarks
                    )

                # Copy current landmarks for next frame
                previous_landmarks = deepcopy(current_landmarks)

                # Draw 33 landmarks
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        thickness=2,
                        circle_radius=3
                    ),
                    mp_draw.DrawingSpec(
                        thickness=2,
                        circle_radius=2
                    )
                )

            else:
                previous_landmarks = None

            # Create and save CSV row
            csv_row = create_csv_row(
                sample_id=sample_id,
                pose_detected=pose_detected,
                movement_score=movement_score,
                landmarks=current_landmarks
            )

            writer.writerow(csv_row)

            # Save data immediately
            csv_file.flush()

            # Display information
            cv2.putText(
                frame,
                f"Sample ID: {sample_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pose Detected: {pose_detected}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Movement Score: {movement_score:.6f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press ESC to stop",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "MediaPipe 33 Landmarks",
                frame
            )

            sample_id += 1

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("ESC pressed. Program বন্ধ হচ্ছে...")
                break

    camera.release()
    pose.close()
    cv2.destroyAllWindows()

    print("============================================")
    print("Data collection completed.")
    print(f"Total saved samples: {sample_id - 1}")
    print(f"CSV saved: {os.path.abspath(OUTPUT_CSV)}")
    print("============================================")


if __name__ == "__main__":
    main()