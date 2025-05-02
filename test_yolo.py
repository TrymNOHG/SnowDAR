from ultralytics import YOLO

# Load a model
model = YOLO("./runs/detect/train14/weights/best.pt")  # pretrained YOLO11n model


model.predict(
    source="/cluster/home/trymhg/comp-vis/SnowDAR/Poles/new_rgb/images/test",
    project="save",
    name=None,
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
    )