# Example
# python test_yolo.py
# python test_yolo.py 5 lidar      # uses train5 and lidar
import os
import argparse
from ultralytics import YOLO

# Set up argument parser
parser = argparse.ArgumentParser(description="Run YOLO model on RGB or LiDAR data.")
parser.add_argument("train_number", nargs="?", default="", help="Train number")
parser.add_argument("data_type", nargs="?", default="rgb", choices=["rgb", "lidar"], help="Data type: rgb or lidar")

# Parse arguments
args = parser.parse_args()

# Paths
model_path = f"./runs/detect/train{args.train_number}/weights/best.pt"
image_path = f"./Poles/new_{args.data_type}/images/test"

os.makedirs("save_runs", exist_ok=True)

# Load and run model
model = YOLO(model_path)
model.predict(
    source=image_path,
    project="test_runs",
    name=None,
    save_txt=True,
    save_conf=True
)
