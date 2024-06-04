import ultralytics
from ultralytics import YOLO
import torchvision
import torch
from roboflow import Roboflow

rf = Roboflow(api_key="6h6vw57qfsq4QeKZWXgR")
project = rf.workspace("atakan-arslan-abioi").project("vacuum-head")
version = project.version(1)
dataset = version.download("yolov8")

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

data_yaml_path ='vacuum-head-1/data.yaml'

model.train(data=data_yaml_path, epochs=2, imgsz=640)
