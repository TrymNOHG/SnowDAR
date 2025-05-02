import os
import torch
from PIL import Image, ImageDraw, ImageFont
from coco_dataset import CocoDataset  
from torchvision.transforms import functional as F
from tqdm import tqdm

def draw_boxes_on_image(img, target, label_map=None):
    draw = ImageDraw.Draw(img)

    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        label_id = int(label.item())
        label_str = str(label_id) if label_map is None else label_map.get(label_id, str(label_id))

        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
        draw.text((x1, y1), label_str, fill="yellow")

    return img

def save_annotated_images(dataset, output_dir, label_map=None):
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc="Saving images"):
        img, target = dataset[i]

        annotated_img = draw_boxes_on_image(img, target, label_map)

        image_id = target["image_id"].item()
        save_path = os.path.join(output_dir, f"{image_id}.jpg")
        annotated_img.save(save_path)
        break

if __name__ == "__main__":
    image_dir = "Poles/rgb"
    annotation_file = "Poles/coco/valid.json"
    output_dir = "./annotated_images"

    dataset = CocoDataset(image_dir, annotation_file, transforms=None)


    label_map = {
        1: "pole"  
    }

    save_annotated_images(dataset, output_dir, label_map)