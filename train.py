
from ultralytics import YOLO
import os


def train_model():

    # -------------------------------
    # Configuration
    # -------------------------------

    DATA_YAML = "dataset/data.yaml"   # Path to data.yaml
    MODEL_TYPE = "yolov8n.pt"         # Base model (nano version)
    EPOCHS = 50
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    PROJECT_NAME = "shoplifting_training"
    RUN_NAME = "yolov8_shoplifting"

    # -------------------------------
    # Load Pretrained YOLOv8 Model
    # -------------------------------

    model = YOLO(MODEL_TYPE)

    # -------------------------------
    # Train Model
    # -------------------------------

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        project=PROJECT_NAME,
        pretrained=True,
        verbose=True
    )

    print("\nTraining Completed Successfully!")
    print("Best weights saved at:")
    print(f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    train_model()