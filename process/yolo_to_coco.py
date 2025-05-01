import os
import json
from PIL import Image

ROOT_DIR = "Poles/rgb"
SPLITS = ["train", "valid", "test"]
CLASS_NAME = "pole"


def convert_split(split_name):
    image_dir = os.path.join(ROOT_DIR, "images", split_name)
    label_dir = os.path.join(ROOT_DIR, "labels", split_name)
    output_json = os.path.join(ROOT_DIR, f"{split_name}.json")

    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    images = []
    annotations = []
    categories = [{"id": 0, "name": CLASS_NAME}]
    ann_id = 0

    for img_id, file_name in enumerate(sorted(image_files)):
        img_path = os.path.join(image_dir, file_name)
        label_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")

        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x, y, w, h = map(float, parts)
                x_min = (x - w / 2) * width
                y_min = (y - h / 2) * height
                bbox_width = w * width
                bbox_height = h * height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"[✓] Saved {split_name}.json with {len(images)} images and {len(annotations)} annotations.")


# Run conversion for all splits
for split in SPLITS:
    convert_split(split)
