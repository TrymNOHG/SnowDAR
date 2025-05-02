import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from coco_dataset import CocoDataset
import os
from tqdm import tqdm
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def evaluate(model, dataset, output_dir, device, conf_threshold=0.5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for img, output, target in zip(images, outputs, targets):
                image_id = target['image_id'].item()
                img_info = dataset.image_info[image_id]
                filename = img_info['file_name'].split("/")[-1]  
                base_name = os.path.splitext(filename)[0]
                txt_path = os.path.join(output_dir, f"{base_name}.txt")

                print(output)

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                lines = []
                for box, score in zip(boxes, scores):
                    if score < conf_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    line = f"0 {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {score:.4f}"
                    lines.append(line)

                with open(txt_path, 'w') as f:
                    f.write('\n'.join(lines))

def get_transform():
    return T.Compose([T.ToTensor()])


if __name__ == "__main__":
    image_dir = "./Poles/rgb/"
    annotation_file = "./Poles/coco/valid.json"
    model_path = "./coco_model2.pt"
    output_dir = "./predictions"
    conf_threshold = 0.001

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = CocoDataset(image_dir, annotation_file, transforms=get_transform())

    num_classes = 1 + 1  


    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    evaluate(model, test_dataset, output_dir, device, conf_threshold)
