
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("path/to/yolov8.pt")  # Replace with your model path

def annotate_image(image: np.ndarray) -> np.ndarray:
    results = model(image)[0]
    annotated = image.copy()
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated
