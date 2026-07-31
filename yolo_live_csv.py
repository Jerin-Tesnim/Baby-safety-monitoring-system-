import os
import csv
import cv2
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = (
    r"C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat"
    r"\runs\detect\baby_person_empty_seat\weights\best.pt"
)

CSV_PATH = (
    r"C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat"
    r"\yolo_live_output.csv"
)

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Increase this value if weak/wrong detections appear.
# Recommended range: 0.50 to 0.80
CONFIDENCE_THRESHOLD = 0.60

# One CSV sample will be saved after every 5 frames.
SAVE_EVERY_N_FRAMES = 5


# Exact class names from the trained model
BABY_CLASS = "baby"
PERSON_CLASS = "person"
EMPTY_SEAT_CLASS = "empty_seat"


# =========================================================
# CSV COLUMNS
# =========================================================

CSV_COLUMNS = [
    "sample_id",
    "detected_class",
    "baby_detected",
    "empty_seat_detected",
    "baby_count",
    "confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area"
]


# =========================================================
# CREATE NEW CSV
# =========================================================

def create_new_csv():
    """
    Creates a new CSV file every time the program starts.
    Previous CSV data will be overwritten.
    Sample ID will start from 1.
    """

    csv_folder = os.path.dirname(CSV_PATH)

    if csv_folder:
        os.makedirs(csv_folder, exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()


# =========================================================
# CLEAN CLASS NAME
# =========================================================

def clean_class_name(class_name):
    """
    Converts a class name to lowercase and removes spaces.
    """

    return str(class_name).strip().lower()


# =========================================================
# FIND CLASS IDS
# =========================================================

def find_class_ids(model):
    """
    Finds the numeric class IDs for baby, person, and empty_seat.
    """

    class_ids = {
        BABY_CLASS: None,
        PERSON_CLASS: None,
        EMPTY_SEAT_CLASS: None
    }

    if isinstance(model.names, dict):
        model_classes = model.names.items()
    else:
        model_classes = enumerate(model.names)

    for class_id, class_name in model_classes:

        cleaned_name = clean_class_name(class_name)

        if cleaned_name in class_ids:
            class_ids[cleaned_name] = int(class_id)

    return class_ids


# =========================================================
# DRAW DETECTION BOX
# =========================================================

def draw_detection_box(
    frame,
    x1,
    y1,
    x2,
    y2,
    class_name,
    confidence
):
    """
    Draws baby, person, and empty_seat boxes.
    """

    if class_name == BABY_CLASS:
        box_color = (0, 255, 0)

    elif class_name == PERSON_CLASS:
        box_color = (255, 150, 0)

    elif class_name == EMPTY_SEAT_CLASS:
        box_color = (255, 0, 255)

    else:
        box_color = (255, 255, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    label = f"{class_name} {confidence:.2f}"

    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2
    )

    text_width, text_height = text_size

    label_y1 = max(0, y1 - text_height - baseline - 8)
    label_y2 = max(text_height + baseline + 8, y1)

    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + text_width + 10, label_y2),
        box_color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y2 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )


# =========================================================
# DRAW SMALL CAMERA INFORMATION
# =========================================================

def draw_camera_information(
    frame,
    sample_id,
    saved_samples,
    detected_class,
    baby_detected,
    empty_seat_detected,
    baby_count
):
    """
    Displays information using text only.
    It does not use a large black rectangle.
    """

    information = [
        f"Sample ID: {sample_id}",
        f"Saved Samples: {saved_samples}",
        f"Detected Class: {detected_class}",
        f"Baby Detected: {baby_detected}",
        f"Empty Seat Detected: {empty_seat_detected}",
        f"Baby Count: {baby_count}",
        "Press Q to stop"
    ]

    start_x = 20
    start_y = 35
    line_gap = 34

    for index, text in enumerate(information):

        text_y = start_y + (index * line_gap)

        # Black text border
        cv2.putText(
            frame,
            text,
            (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4
        )

        # White visible text
        cv2.putText(
            frame,
            text,
            (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    # -----------------------------------------------------
    # Check model file
    # -----------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("=" * 70)
        print("ERROR: best.pt file was not found.")
        print("Check the following model path:")
        print(MODEL_PATH)
        print("=" * 70)

        return

    # -----------------------------------------------------
    # Load YOLO model
    # -----------------------------------------------------

    print("Loading YOLO model...")

    try:
        model = YOLO(MODEL_PATH)

    except Exception as error:
        print("YOLO model loading error:")
        print(error)
        return

    print("YOLO model loaded successfully.")

    # -----------------------------------------------------
    # Display model classes
    # -----------------------------------------------------

    print("\nModel class names:")

    if isinstance(model.names, dict):

        for class_id, class_name in model.names.items():
            print(f"{class_id}: {class_name}")

    else:

        for class_id, class_name in enumerate(model.names):
            print(f"{class_id}: {class_name}")

    # -----------------------------------------------------
    # Find required class IDs
    # -----------------------------------------------------

    class_ids = find_class_ids(model)

    baby_class_id = class_ids[BABY_CLASS]
    person_class_id = class_ids[PERSON_CLASS]
    empty_seat_class_id = class_ids[EMPTY_SEAT_CLASS]

    missing_classes = []

    if baby_class_id is None:
        missing_classes.append(BABY_CLASS)

    if person_class_id is None:
        missing_classes.append(PERSON_CLASS)

    if empty_seat_class_id is None:
        missing_classes.append(EMPTY_SEAT_CLASS)

    if missing_classes:

        print("\nERROR: These classes were not found in the model:")
        print(", ".join(missing_classes))
        print("\nThe model contains:")
        print(model.names)

        return

    print("\nDetected model class IDs:")
    print("Baby class ID      :", baby_class_id)
    print("Person class ID    :", person_class_id)
    print("Empty seat class ID:", empty_seat_class_id)

    # -----------------------------------------------------
    # Create a completely new CSV
    # -----------------------------------------------------

    try:
        create_new_csv()

    except PermissionError:

        print("\nERROR: yolo_live_output.csv is open.")
        print("Close the CSV file from Excel and run the code again.")

        return

    except Exception as error:

        print("\nCSV creation error:")
        print(error)

        return

    print("\nNew CSV file created.")
    print("Sample ID will start from 1.")
    print("CSV path:")
    print(CSV_PATH)

    # -----------------------------------------------------
    # Open camera
    # -----------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not camera.isOpened():

        print("\nERROR: Camera could not be opened.")
        print("Change CAMERA_INDEX from 0 to 1 if needed.")

        return

    print("\nCamera started successfully.")
    print("Press Q to close the camera.")
    print("=" * 70)

    sample_id = 1
    frame_number = 0
    total_saved_samples = 0

    total_baby_detections = 0
    total_person_detections = 0
    total_empty_seat_detections = 0

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the camera.")
                break

            frame_number += 1

            display_frame = frame.copy()

            # -------------------------------------------------
            # Run YOLO prediction for all three classes
            # -------------------------------------------------

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[
                    baby_class_id,
                    person_class_id,
                    empty_seat_class_id
                ],
                verbose=False
            )

            result = results[0]

            baby_detections = []
            person_detections = []
            empty_seat_detections = []

            # -------------------------------------------------
            # Read all detected objects
            # -------------------------------------------------

            if result.boxes is not None and len(result.boxes) > 0:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    class_name = clean_class_name(
                        model.names[class_id]
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    x1_float, y1_float, x2_float, y2_float = (
                        box.xyxy[0].tolist()
                    )

                    bbox_width = max(
                        0.0,
                        x2_float - x1_float
                    )

                    bbox_height = max(
                        0.0,
                        y2_float - y1_float
                    )

                    bbox_area = (
                        bbox_width * bbox_height
                    )

                    detection_information = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": x1_float,
                        "y1": y1_float,
                        "x2": x2_float,
                        "y2": y2_float,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_area": bbox_area
                    }

                    if class_name == BABY_CLASS:

                        baby_detections.append(
                            detection_information
                        )

                    elif class_name == PERSON_CLASS:

                        person_detections.append(
                            detection_information
                        )

                    elif class_name == EMPTY_SEAT_CLASS:

                        empty_seat_detections.append(
                            detection_information
                        )

                    # Draw every valid detection
                    draw_detection_box(
                        display_frame,
                        int(x1_float),
                        int(y1_float),
                        int(x2_float),
                        int(y2_float),
                        class_name,
                        confidence
                    )

            # -------------------------------------------------
            # Detection flags and counts
            # -------------------------------------------------

            baby_count = len(baby_detections)

            person_count = len(person_detections)

            empty_seat_count = len(
                empty_seat_detections
            )

            baby_detected = (
                1 if baby_count > 0 else 0
            )

            empty_seat_detected = (
                1 if empty_seat_count > 0 else 0
            )

            # -------------------------------------------------
            # Select detected_class
            #
            # Priority:
            # 1. If baby exists       -> baby
            # 2. Otherwise person     -> person
            # 3. Otherwise empty seat -> empty_seat
            # 4. Otherwise            -> none
            # -------------------------------------------------

            selected_detection = None

            if baby_count > 0:

                detected_class = BABY_CLASS

                selected_detection = max(
                    baby_detections,
                    key=lambda item: item["confidence"]
                )

            elif person_count > 0:

                detected_class = PERSON_CLASS

                selected_detection = max(
                    person_detections,
                    key=lambda item: item["confidence"]
                )

            elif empty_seat_count > 0:

                detected_class = EMPTY_SEAT_CLASS

                selected_detection = max(
                    empty_seat_detections,
                    key=lambda item: item["confidence"]
                )

            else:

                detected_class = "none"

            # -------------------------------------------------
            # Selected detection information
            # -------------------------------------------------

            if selected_detection is not None:

                selected_confidence = (
                    selected_detection["confidence"]
                )

                selected_bbox_width = (
                    selected_detection["bbox_width"]
                )

                selected_bbox_height = (
                    selected_detection["bbox_height"]
                )

                selected_bbox_area = (
                    selected_detection["bbox_area"]
                )

            else:

                selected_confidence = 0.0
                selected_bbox_width = 0.0
                selected_bbox_height = 0.0
                selected_bbox_area = 0.0

            # -------------------------------------------------
            # Save one row after every N frames
            # -------------------------------------------------

            if frame_number % SAVE_EVERY_N_FRAMES == 0:

                csv_row = {
                    "sample_id": sample_id,
                    "detected_class": detected_class,
                    "baby_detected": baby_detected,
                    "empty_seat_detected": empty_seat_detected,
                    "baby_count": baby_count,
                    "confidence": round(
                        selected_confidence,
                        4
                    ),
                    "bbox_width": round(
                        selected_bbox_width,
                        2
                    ),
                    "bbox_height": round(
                        selected_bbox_height,
                        2
                    ),
                    "bbox_area": round(
                        selected_bbox_area,
                        2
                    )
                }

                try:

                    with open(
                        CSV_PATH,
                        "a",
                        newline="",
                        encoding="utf-8"
                    ) as file:

                        writer = csv.DictWriter(
                            file,
                            fieldnames=CSV_COLUMNS
                        )

                        writer.writerow(csv_row)

                    total_saved_samples += 1

                    total_baby_detections += (
                        baby_count
                    )

                    total_person_detections += (
                        person_count
                    )

                    total_empty_seat_detections += (
                        empty_seat_count
                    )

                    sample_id += 1

                except PermissionError:

                    print(
                        "\nERROR: CSV file is open in Excel."
                    )

                    print(
                        "Close the CSV file and restart the program."
                    )

                    break

                except Exception as error:

                    print("\nCSV save error:")
                    print(error)

                    break

            # -------------------------------------------------
            # Show small information on camera
            # -------------------------------------------------

            draw_camera_information(
                display_frame,
                sample_id,
                total_saved_samples,
                detected_class,
                baby_detected,
                empty_seat_detected,
                baby_count
            )

            cv2.imshow(
                "YOLO Baby Person Empty Seat Detection",
                display_frame
            )

            pressed_key = cv2.waitKey(1) & 0xFF

            if (
                pressed_key == ord("q")
                or pressed_key == ord("Q")
            ):

                print("\nQ pressed. Camera is closing.")
                break

    except KeyboardInterrupt:

        print("\nProgram stopped from the keyboard.")

    finally:

        camera.release()
        cv2.destroyAllWindows()

        last_saved_sample_id = sample_id - 1

        print("\n" + "=" * 70)
        print("YOLO LIVE DATA COLLECTION FINISHED")
        print("=" * 70)
        print("Starting Sample ID        : 1")
        print(
            "Last Saved Sample ID      :",
            last_saved_sample_id
        )
        print(
            "New Samples Saved         :",
            total_saved_samples
        )
        print(
            "Baby Detections           :",
            total_baby_detections
        )
        print(
            "Person Detections         :",
            total_person_detections
        )
        print(
            "Empty Seat Detections     :",
            total_empty_seat_detections
        )
        print(
            "CSV Saved At              :",
            CSV_PATH
        )
        print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()