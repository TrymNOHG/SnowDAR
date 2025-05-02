import os
import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import torchvision.transforms as T

class CocoDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.image_info = {img['id']: img for img in coco['images']}
        self.annotations = coco['annotations']
        self.image_ids = list(self.image_info.keys())


        self.imgToAnns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []
            self.imgToAnns[img_id].append(ann)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.image_info[image_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        annots = self.imgToAnns.get(image_id, [])
        boxes = []
        labels = []
        for ann in annots:
            boxes.append(ann['bbox']) 
            labels.append(ann.get('category_id', 1))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
            boxes[:, 2:] += boxes[:, :2] 
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)


        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)
