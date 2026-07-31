from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=r"C:\Users\Admin\Desktop\YOLO_Baby_Person_Seat\Baby person empty seat detection.v2i.yolov8\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=2,
    project="runs",
    name="baby_person_empty_seat",
    exist_ok=True
)

print("Training completed successfully.")