import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from coco_dataset import CocoDataset
import torchvision.transforms.functional as F
from PIL import ImageDraw

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform():
    return T.Compose([T.ToTensor()])

image_dir = "Poles/rgb"

train_dataset = CocoDataset(image_dir, 'Poles/coco/train.json', get_transform())
valid_dataset = CocoDataset(image_dir, 'Poles/coco/valid.json', get_transform())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


num_classes = 2  # background + 1 for pole object
model = get_model(num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for imgs, targets in train_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
    
    # Just for testing whether my train is working
    model.eval()
    img, _ = train_dataset[0]  
    img_tensor = img.to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]

    img_pil = F.to_pil_image(img)

    draw = ImageDraw.Draw(img_pil)
    boxes = output['boxes'].cpu()
    scores = output['scores'].cpu()

    num_detections = 0
    for box, score in zip(boxes, scores):
        if score < 0.05:
            continue
        num_detections += 1
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{score:.2f}", fill="yellow")

    img_pil.save(f"epoch_{epoch}_check.jpg")
    print(f"[Epoch {epoch}] Detected {num_detections} boxes on training image.")


print(f"Epoch {epoch} loss: {losses.item():.4f}")

torch.save(model.state_dict(), "coco_model2.pt")