import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import os

# === Paths ===
root_img = "Poles/rgb/images"
ann_dir = "Poles/rgb"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset wrapper ===
class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
        self.transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])
        self.min_box_size = 2  # minimum width/height in pixels

    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        img = self.transforms(img)

        boxes = []
        labels = []

        for obj in targets:
            x, y, w, h = obj["bbox"]
            if w < self.min_box_size or h < self.min_box_size:
                continue  # skip tiny or invalid boxes

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])
            labels.append(obj["category_id"])

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))  # skip if no valid targets

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target

# === Load datasets ===
train_dataset = CocoDataset(f"{root_img}/train", f"{ann_dir}/train.json")
valid_dataset = CocoDataset(f"{root_img}/valid", f"{ann_dir}/valid.json")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === Load RetinaNet with COCO weights ===
weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
model = retinanet_resnet50_fpn(weights=weights)

# === Replace classifier for 1 class (pole) + background
num_classes = 2
model.head.classification_head.num_classes = num_classes
model.head.classification_head.cls_logits = torch.nn.Conv2d(
    256, num_classes * 9, kernel_size=3, stride=1, padding=1
)

model.to(device)

# === Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

# === Training loop
for epoch in range(10):
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batch if any image has no boxes
        if any(len(t["boxes"]) == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            print("Warning: non-finite loss, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

print("✅ Done!")
