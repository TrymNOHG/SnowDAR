# TODO: Visual bounding box
import os
import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import ToImage, ToDtype, Compose
from PIL import Image
from torchvision.ops import nms

# === Settings ===
image_dir = "Poles/rgb/images/test"
output_file = "retinanet_preds_yolo.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf_thresh = 0.5
iou_thresh = 0.4

# === Load model ===
model.eval()
model.to(device)

# === Preprocessing ===
transform = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),
])

# === Run inference ===
with open(output_file, "w") as out:
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor)[0]

        boxes = preds["boxes"]
        scores = preds["scores"]
        labels = preds["labels"]

        keep = (scores > conf_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.numel() == 0:
            continue

        # NMS
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep_idx]

        # Convert to YOLO format
        for box in boxes:
            x1, y1, x2, y2 = box.cpu()
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            out.write(f"{fname} 0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
