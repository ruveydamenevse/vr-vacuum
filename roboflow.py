import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import supervision as sv
import numpy as np
from PIL import Image, ImageDraw

# Define image path and model details
image_path = "hard.png"
MODEL_ID = "vacuum-head/4"
#MODEL_ID = "vacuum-detection-bdjga/1"


# Set up the model client with API details and configuration
config = InferenceConfiguration(confidence_threshold=0.2, iou_threshold=0.2)
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="6h6vw57qfsq4QeKZWXgR",
)
client.configure(config)
client.select_model(MODEL_ID)

# Perform inference on the image
predictions = client.infer(image_path)
print(predictions)

# Assuming predictions is structured correctly and contains bounding box data
# We need to extract bounding box data from predictions
bounding_boxes = predictions['predictions']

# Load image using cv2
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Unable to find the image '{image_path}'")

# Prepare data for supervision
xyxy = np.array([[pred['x'] - pred['width']/2, pred['y']-pred['height']/2, pred['x'] + pred['width']/2, pred['y'] + pred['height']/2] for pred in bounding_boxes])
class_ids = np.array([pred['class_id'] for pred in bounding_boxes])
confidences = np.array([pred['confidence'] for pred in bounding_boxes])

# Create Detections object for supervision
detections = sv.Detections(
    xyxy=xyxy,
    class_id=class_ids,
    confidence=confidences
)

# Annotate the image using supervision
bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# Display the annotated image
sv.plot_image(annotated_frame)
