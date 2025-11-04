import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    box1 = box1.clone()
    box2 = box2.clone()
    x1_min = float(box1[0] - box1[2] / 2)
    y1_min = float(box1[1] - box1[3] / 2)
    x1_max = float(box1[0] + box1[2] / 2)
    y1_max = float(box1[1] + box1[3] / 2)
    x2_min = float(box2[0] - box2[2] / 2)
    y2_min = float(box2[1] - box2[3] / 2)
    x2_max = float(box2[0] + box2[2] / 2)
    y2_max = float(box2[1] + box2[3] / 2)
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

# Load model
model_path = os.getenv('BEST_MODEL_PATH')
model = YOLO(model_path)

# Val images dir
val_images_dir = Path('checkpoints/yolo_dataset/val/images')

all_ious = []
for img_file in val_images_dir.glob('*.'):
    label_file = val_images_dir.parent / 'labels' / (img_file.stem + '.txt')
    gt_by_class = {}
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    gt_by_class[cid] = torch.tensor([cx, cy, w, h], dtype=torch.float32)

    preds = model.predict(str(img_file), conf=0.25, verbose=False)
    pred_by_class = {}
    if len(preds) > 0 and preds[0].boxes is not None:
        boxes = preds[0].boxes
        for box in boxes:
            cls = int(box.cls.item())
            xywh = box.xywhn[0]  # normalized
            if cls not in pred_by_class:
                pred_by_class[cls] = []
            pred_by_class[cls].append(xywh)

    # Assuming one pred per class
    classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))
    ious = []
    for c in classes:
        if c in gt_by_class and c in pred_by_class and pred_by_class[c]:
            gt_box = gt_by_class[c]
            pred_box = pred_by_class[c][0]  # take first
            ious.append(box_iou(gt_box, pred_box))
    if ious:
        all_ious.append(float(np.mean(ious)))

mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
print(f"Mean IoU on validation: {mean_iou:.4f}")
