
import io
import cv2
import base64
import numpy as np

def encode_image_to_jpeg(image: np.ndarray) -> bytes:
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()

def image_to_base64(image: np.ndarray) -> str:
    jpeg_bytes = encode_image_to_jpeg(image)
    return base64.b64encode(jpeg_bytes).decode("utf-8")