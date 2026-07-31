import os
import cv2

OUTPUT_DIR = "captured_images"
CAMERA_INDEX = 0
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

START_IMAGE_NUMBER = 133


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_number = START_IMAGE_NUMBER

    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Camera started successfully.")
    print(f"Images will be saved from image_{image_number}.jpg")
    print("Press S to save an image.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read a frame.")
            break

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Next Image: image_{image_number}.jpg",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            "Press S to Save | Press Q to Quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Dataset Image Capture", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = f"image_{image_number}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")

            image_number += 1

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()