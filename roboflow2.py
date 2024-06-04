import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import supervision as sv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bounding_boxes(image, boxes):
    # Iterate through all the bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        # Draw the rectangle on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 7)
    return image

def detect_objects_in_image(image_array):
    MODEL_ID = "vacuum-detection-bdjga/4"  # Model ID'nizi buraya yazın
    config = InferenceConfiguration(confidence_threshold=0.2, iou_threshold=0.2)
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="6h6vw57qfsq4QeKZWXgR",  # API anahtarınızı buraya girin
    )
    client.configure(config)
    client.select_model(MODEL_ID)

    # Perform inference on the image array
    predictions = client.infer(image_array)
    print("Raw predictions:", predictions)

    bounding_boxes = predictions['predictions']
    if not bounding_boxes:
        print("No objects detected.")
        return None

    # Extract bounding box coordinates
    xyxy = np.array([[pred['x'] - pred['width']/2, pred['y'] - pred['height']/2, pred['x'] + pred['width']/2, pred['y'] + pred['height']/2] for pred in bounding_boxes])

    # Print coordinates for each detection
    print("Detected objects coordinates:")
    for coords in xyxy:
        print(f"Coordinates (x1, y1, x2, y2): {coords}")



    return xyxy

