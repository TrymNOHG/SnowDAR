import os
import numpy as np
from glob import glob


class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height, confidence=None):
        self.class_id = int(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)
        self.confidence = float(confidence) if confidence is not None else None
        
    @property
    def x_min(self):
        return self.x_center - self.width / 2
    
    @property
    def y_min(self):
        return self.y_center - self.height / 2
    
    @property
    def x_max(self):
        return self.x_center + self.width / 2
    
    @property
    def y_max(self):
        return self.y_center + self.height / 2
    
    @property
    def area(self):
        return self.width * self.height


def read_yolo_file(filepath, is_prediction=False):
    if not os.path.exists(filepath):
        return []
    
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Skip empty lines
            if not parts:
                continue
                
            if is_prediction and len(parts) >= 6:
                # Prediction format: class_id x_center y_center width height confidence
                class_id, x_center, y_center, width, height, confidence = parts[:6]
                box = BoundingBox(class_id, x_center, y_center, width, height, confidence)
            elif not is_prediction and len(parts) >= 5:
                # Ground truth format: class_id x_center y_center width height
                class_id, x_center, y_center, width, height = parts[:5]
                box = BoundingBox(class_id, x_center, y_center, width, height)
            else:
                print(f"Warning: Invalid format in line: {line}")
                continue
                
            boxes.append(box)
    
    return boxes


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x_min = max(box1.x_min, box2.x_min)
    y_min = max(box1.y_min, box2.y_min)
    x_max = min(box1.x_max, box2.x_max)
    y_max = min(box1.y_max, box2.y_max)
    
    # Check if boxes overlap
    if x_max < x_min or y_max < y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    union = box1.area + box2.area - intersection
    
    # Avoid division by zero
    if union <= 0:
        return 0.0
        
    return intersection / union


def calculate_precision_recall_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculate precision, recall, and average precision for a single class
    at a specified IoU threshold
    """
    # Sort predictions by confidence in descending order
    pred_boxes = sorted(pred_boxes, key=lambda x: x.confidence, reverse=True)
    
    # Create binary array to track which ground truth boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    
    # Arrays to store precision and recall values
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    # Process each prediction
    for i, pred in enumerate(pred_boxes):
        # Find the best matching ground truth box
        best_iou = iou_threshold  # Only consider matches with IoU ≥ threshold
        best_gt_idx = -1
        
        for j, gt in enumerate(gt_boxes):
            if gt.class_id == pred.class_id and not gt_matched[j]:
                iou = calculate_iou(pred, gt)
                if iou >= best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # Check if a valid match was found
        if best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            tp[i] = 1  # True positive
        else:
            fp[i] = 1  # False positive
    
    # Calculate cumulative sums for true positives and false positives
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    
    # Calculate precision and recall at each threshold
    precision = np.zeros(len(pred_boxes))
    recall = np.zeros(len(pred_boxes))
    
    for i in range(len(pred_boxes)):
        if cumsum_tp[i] + cumsum_fp[i] > 0:
            precision[i] = cumsum_tp[i] / (cumsum_tp[i] + cumsum_fp[i])
        if len(gt_boxes) > 0:
            recall[i] = cumsum_tp[i] / len(gt_boxes)
    
    # If there are no predictions or ground truths, return 0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate final precision and recall
    final_precision = precision[-1] if len(precision) > 0 else 0
    final_recall = recall[-1] if len(recall) > 0 else 0
    
    # Calculate AP using the 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return final_precision, final_recall, ap


def calculate_map(gt_files, pred_files, iou_thresholds=[0.5]):
    """
    Calculate mAP across all classes at a specific IoU threshold or range of thresholds
    """
    all_classes = set()
    file_pairs = []
    
    # Find pairs of ground truth and prediction files
    for gt_file in gt_files:
        base_name = os.path.basename(gt_file)
        image_id = os.path.splitext(base_name)[0]
        
        # Find corresponding prediction file if it exists
        pred_file_candidates = [p for p in pred_files if os.path.splitext(os.path.basename(p))[0] == image_id]
        
        if pred_file_candidates:
            pred_file = pred_file_candidates[0]
        else:
            # No prediction file means no detections for this image
            pred_file = None
        
        file_pairs.append((gt_file, pred_file))
        
        # Collect all classes present in ground truth
        gt_boxes = read_yolo_file(gt_file, is_prediction=False)
        for box in gt_boxes:
            all_classes.add(box.class_id)
    
    # Organize all boxes by class
    class_gt_boxes = {cls: [] for cls in all_classes}
    class_pred_boxes = {cls: [] for cls in all_classes}
    
    # Read all files and organize boxes by class
    for gt_file, pred_file in file_pairs:
        gt_boxes = read_yolo_file(gt_file, is_prediction=False)
        pred_boxes = [] if pred_file is None else read_yolo_file(pred_file, is_prediction=True)
        
        # Add image_id to keep track of which image each box belongs to
        image_id = os.path.splitext(os.path.basename(gt_file))[0]
        
        for box in gt_boxes:
            box.image_id = image_id
            class_gt_boxes[box.class_id].append(box)
        
        for box in pred_boxes:
            box.image_id = image_id
            if box.class_id in class_pred_boxes:  # Only include classes that exist in ground truth
                class_pred_boxes[box.class_id].append(box)
    
    # Calculate metrics for each IoU threshold
    results = {}
    for iou_threshold in iou_thresholds:
        class_metrics = {}
        
        for class_id in all_classes:
            precision, recall, ap = calculate_precision_recall_ap(
                class_gt_boxes[class_id], 
                class_pred_boxes[class_id], 
                iou_threshold
            )
            class_metrics[class_id] = {
                'precision': precision,
                'recall': recall,
                'ap': ap
            }
        
        # Calculate mean metrics across all classes
        mean_precision = np.mean([m['precision'] for m in class_metrics.values()])
        mean_recall = np.mean([m['recall'] for m in class_metrics.values()])
        mean_ap = np.mean([m['ap'] for m in class_metrics.values()])
        
        results[iou_threshold] = {
            'class_metrics': class_metrics,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mAP': mean_ap
        }
    
    return results


def main():
    gt_dir = 'valid_labels'  # Directory containing ground truth files
    pred_dir = 'valid_predictions'  # Directory containing prediction files
    
    # Get file lists
    gt_files = glob(os.path.join(gt_dir, '*.txt'))
    pred_files = glob(os.path.join(pred_dir, '*.txt'))
    
    print(f"Found {len(gt_files)} ground truth files and {len(pred_files)} prediction files")
    
    # Calculate mAP@50
    map50_results = calculate_map(gt_files, pred_files, [0.5])
    
    # Calculate mAP@50:95 
    map_50_95_thresholds = np.linspace(0.5, 0.95, 10)
    map_50_95_results = calculate_map(gt_files, pred_files, map_50_95_thresholds)
    
    # Print results
    print("\n=== Results ===")
    
    map50 = map50_results[0.5]['mAP']
    precision50 = map50_results[0.5]['mean_precision']
    recall50 = map50_results[0.5]['mean_recall']
    
    print(f"\nAt IoU=0.50:")
    print(f"Precision: {precision50:.4f}")
    print(f"Recall: {recall50:.4f}")
    print(f"mAP@50: {map50:.4f}")
    
    # mAP@50:95
    map_50_95 = np.mean([results['mAP'] for _, results in map_50_95_results.items()])
    print(f"\nmAP@50:95: {map_50_95:.4f}")


if __name__ == "__main__":
    main()
