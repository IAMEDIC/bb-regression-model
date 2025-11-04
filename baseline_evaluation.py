import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import itertools

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from mlflow.data.dataset import Dataset as MLFLowDataset
from torch.utils.data import Dataset
from ultralytics import YOLO

# Load environment variables
load_dotenv()
np.random.seed(0)

# Configuration
TARGET_HEIGHT = 640
TARGET_WIDTH = 640
FINETUNE_DS_DIR = os.getenv("FINETUNE_DIR")
DATASET_IMAGES_DIR_NAME = "media"
DATASET_IMAGES_DIR = os.path.join(FINETUNE_DS_DIR, DATASET_IMAGES_DIR_NAME)
ANNOTATIONS_PATH = os.path.join(FINETUNE_DS_DIR, "picture_bb_annotations.csv")
IMG_METADATA_PATH = os.path.join(FINETUNE_DS_DIR, "media.csv")
DATASET_VERSION = "2025-30-09"
BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH")
MLFLOW_URI = os.getenv("MLFLOW_URI")
MLFLOW_EXPERIMENT_NAME = f"Fetal_Structures_yolo_{TARGET_HEIGHT}x{TARGET_WIDTH}_baseline_evaluation"
MLFLOW_USER = os.getenv("MLFLOW_USER")
MODEL_NAME = f"fetal_structures_yolo_{TARGET_HEIGHT}x{TARGET_WIDTH}_baseline_{DATASET_VERSION}"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_and_preprocess_data():
    """Load and preprocess annotations and metadata."""
    img_metadata_df = pd.read_csv(IMG_METADATA_PATH)
    img_metadata_df = img_metadata_df[img_metadata_df['media_type'] == 'frame']

    annotations_df = pd.read_csv(ANNOTATIONS_PATH)
    annotations_df = annotations_df[annotations_df['bb_class'].isin(['NB', 'NT'])]
    annotations_df = annotations_df[annotations_df['media_type'] == 'frame']

    # Merge with metadata
    annotations_df = annotations_df.merge(
        img_metadata_df[['id', 'file_path', 'filename']],
        left_on='media_id',
        right_on='id',
        suffixes=('', '_media')
    )

    # Convert to int
    annotations_df['x_min'] = annotations_df['x_min'].astype(int)
    annotations_df['y_min'] = annotations_df['y_min'].astype(int)
    annotations_df['width'] = annotations_df['width'].astype(int)
    annotations_df['height'] = annotations_df['height'].astype(int)

    return annotations_df


def create_full_dataset_split(annotations_df):
    """Create a full dataset split (all images as validation for baseline)."""
    unique_img_paths = annotations_df['file_path'].unique()
    val_img_names = unique_img_paths.tolist()

    # Create annotations dict
    ANNOTATIONS = {}
    class_to_id = {'NB': 0, 'NT': 1}
    for _, row in annotations_df.iterrows():
        fname = row['file_path']
        if fname not in ANNOTATIONS:
            ANNOTATIONS[fname] = []
        class_id = class_to_id[row['bb_class']]
        xmin, ymin, width, height = row['x_min'], row['y_min'], row['width'], row['height']
        xmax, ymax = xmin + width, ymin + height
        box = (xmin, ymin, xmax, ymax)
        ANNOTATIONS[fname].append((class_id, box))

    print(f"Baseline evaluation images: {len(val_img_names)}")

    return val_img_names, ANNOTATIONS


def get_transforms():
    """Define data augmentation transforms for validation (no augmentation)."""
    val_transform = A.Compose([
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})

    return val_transform


class FinetuneDetectionDataset(Dataset):
    def __init__(self, image_root: str, img_names: list[str], annotations: dict[str, list[tuple[int, tuple[int,int,int,int]]]], transform: A.Compose | None = None):
        self.image_root = image_root
        self.img_names = img_names
        self.annotations = annotations
        self.transform = transform
        class_ids = set()
        for boxes in annotations.values():
            for cid, _ in boxes:
                class_ids.add(int(cid))
        self.class_to_idx = {('NB' if i == 0 else 'NT'): i for i in sorted(class_ids)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.img_names)

    def _resolve_image_path(self, rel_path_no_ext: str) -> str:
        candidate = Path(self.image_root) / rel_path_no_ext
        if candidate.suffix:
            if candidate.exists():
                return str(candidate)
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in exts:
            p = Path(self.image_root) / f"{rel_path_no_ext}{ext}"
            if p.exists():
                return str(p)
        p_no_ext = Path(self.image_root) / rel_path_no_ext
        if p_no_ext.parent.exists():
            stem = p_no_ext.name
            for file in p_no_ext.parent.iterdir():
                if file.is_file() and file.stem == stem:
                    return str(file)
        return str(Path(self.image_root) / f"{rel_path_no_ext}.jpg")

    def __getitem__(self, idx: int):
        img_key = self.img_names[idx]
        img_path = self._resolve_image_path(img_key)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pairs = self.annotations.get(img_key, [])
        bboxes = []
        labels = []
        for cid, (xmin, ymin, xmax, ymax) in pairs:
            # Ensure valid box
            if xmax <= xmin or ymax <= ymin:
                continue
            bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            labels.append(int(cid))

        class_labels = labels.copy()

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            labels = class_labels

        image = np.asarray(image)
        img_name = Path(img_key).name
        if '.' in img_name:
            img_name = img_name.rsplit('.', 1)[0]

        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'class_labels': class_labels,
            'img_name': img_name,
            'img_path': img_path,
        }


class YOLODatasetPreparator:
    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = Path(output_dir)

    def prepare_split(self, split_name):
        split_dir = self.output_dir / split_name
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing {split_name} split with {len(self.dataset)} images...")

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            image = sample['image']
            bboxes = sample['bboxes']
            labels = sample['labels']
            img_name = sample['img_name']

            if len(bboxes) == 0:
                print(f"  Warning: Skipping image {img_name} (no bboxes after augmentation)")
                continue

            image_path = images_dir / f"{img_name}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            label_path = labels_dir / f"{img_name}.txt"
            with open(label_path, 'w') as f:
                img_h, img_w = image.shape[:2]

                for bbox, label in zip(bboxes, labels):
                    x_min, y_min, x_max, y_max = bbox

                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    x_center = ((x_min + x_max) / 2) / img_w
                    y_center = ((y_min + y_max) / 2) / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h

                    # Write: <class_id> <x_center> <y_center> <width> <height>
                    f.write(f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(self.dataset)} images")

        print(f"✅ {split_name} split prepared: {len(list(images_dir.glob('*.jpg')))} images")
        return split_dir


class ImageListDataset(MLFLowDataset):
    def __init__(self, names: list[str], source: str = "picture_bb_annotations.csv", version: str = DATASET_VERSION):
        self._names = names
        self._source = source
        self._version = version

    def to_dict(self):
        return {
            "name": "image_list_dataset",
            "digest": hashlib.md5(",".join(self._names).encode()).hexdigest(),
            "source_type": "inline",
            "source": self._source,
            "schema": None,
            "profile": json.dumps({
                "version": self._version,
                "num_images": len(self._names),
                "filenames": self._names
            }),
        }


def create_yolo_config(dataset_dir, train_dataset, output_path=None):
    if output_path is None:
        output_path = Path(dataset_dir) / 'dataset.yaml'

    config = {
        'path': str(Path(dataset_dir).absolute()),
        'train': 'train/images',  # Not used for baseline
        'val': 'val/images',
        'nc': len(train_dataset.class_to_idx),
        'names': list(train_dataset.class_to_idx.keys())
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ YOLO config created at: {output_path}")
    return output_path


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
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {MLFLOW_URI}")
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")


def compute_baseline_metrics(model, yolo_config_path, YOLO_DATASET_DIR, prefix="baseline"):
    """Compute baseline metrics including mAP@50 and custom box IOU"""
    val_results = model.val(data=str(yolo_config_path), split='val', verbose=True, plots=True)

    val_map50 = 0.0
    if hasattr(val_results, 'results_dict'):
        for k, v in val_results.results_dict.items():
            if isinstance(v, (int, float)):
                safe_k = str(k).replace('(', '_').replace(')', '').replace('[', '_').replace(']', '').replace(' ', '_')
                mlflow.log_metric(f"{prefix}/{safe_k}", float(v))
                if 'mAP50_B' in safe_k:
                    val_map50 = float(v)

    # Custom IOU evaluation
    all_ious = []
    val_images_dir = YOLO_DATASET_DIR / 'val' / 'images'
    for img_file in list(val_images_dir.glob('*.jpg')):
        label_file = val_images_dir.parent / 'labels' / (img_file.stem + '.txt')
        gt_by_class = {}
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid_token = parts[0]
                        try:
                            cid = int(cid_token)
                        except ValueError:
                            cid = int(float(cid_token))
                        cx, cy, w, h = map(float, parts[1:])
                        gt_by_class[cid] = torch.tensor([cx, cy, w, h], dtype=torch.float32)

        preds = model.predict(str(img_file), conf=0.25, verbose=False)
        pred_by_class = {}
        if len(preds) > 0 and preds[0].boxes is not None:
            boxes = preds[0].boxes
            if hasattr(boxes, 'xywhn'):
                xywhn = boxes.xywhn.cpu().numpy()
            else:
                img = cv2.imread(str(img_file))
                h, w = img.shape[:2]
                xyxy = boxes.xyxy.cpu().numpy()
                xywhn = []
                for x1, y1, x2, y2 in xyxy:
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    ww = (x2 - x1) / w
                    hh = (y2 - y1) / h
                    xywhn.append([cx, cy, ww, hh])
                xywhn = np.array(xywhn)
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            for i, c in enumerate(cls):
                if c not in pred_by_class or conf[i] > pred_by_class[c][1]:
                    pred_by_class[c] = (torch.tensor(xywhn[i], dtype=torch.float32), float(conf[i]))

        classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))
        if not classes:
            continue

        gt_mask = torch.zeros(2)
        pred_mask = torch.zeros(2)
        gt_boxes = torch.zeros((2, 4))
        pred_boxes = torch.zeros((2, 4))
        for c in classes:
            if c in gt_by_class:
                gt_mask[c] = 1
                gt_boxes[c] = gt_by_class[c]
            if c in pred_by_class:
                pred_mask[c] = 1
                pred_boxes[c] = pred_by_class[c][0]

        ious = []
        for c in classes:
            if gt_mask[c] == 1 and pred_mask[c] == 1:
                ious.append(box_iou(gt_boxes[c], pred_boxes[c]))
        if ious:
            all_ious.append(float(np.mean(ious)))

    val_box_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mlflow.log_metric(f'{prefix}_box_iou_mean', val_box_iou)

    return val_map50, val_box_iou


def main():
    """Main function to run baseline evaluation"""

    # Setup
    setup_mlflow()
    annotations_df = load_and_preprocess_data()

    # Create full dataset split (all as val)
    val_img_names, ANNOTATIONS = create_full_dataset_split(annotations_df)

    # Get transforms (validation only)
    val_transform = get_transforms()

    # Create dataset
    val_dataset = FinetuneDetectionDataset(
        image_root=DATASET_IMAGES_DIR,
        img_names=val_img_names,
        annotations=ANNOTATIONS,
        transform=val_transform
    )

    # Prepare YOLO dataset
    BASELINE_RUN_DIR = Path('runs') / 'baseline'
    YOLO_DATASET_DIR = BASELINE_RUN_DIR / 'dataset'
    YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    val_prep = YOLODatasetPreparator(val_dataset, YOLO_DATASET_DIR)
    val_prep.prepare_split('val')

    yolo_config_path = create_yolo_config(YOLO_DATASET_DIR, val_dataset)

    # Load base model
    model_ckpt = BEST_MODEL_PATH if BEST_MODEL_PATH and os.path.exists(BEST_MODEL_PATH) else 'yolov8n.pt'
    print(f'Loading baseline model checkpoint: {model_ckpt}')
    model = YOLO(model_ckpt)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{MLFLOW_USER}_baseline_{Path(model_ckpt).stem}") as run:
        mlflow_run_id = run.info.run_id

        # Log hyperparameters
        params_to_log = {
            'model_type': 'YOLOv8',
            'model_checkpoint': model_ckpt,
            'num_classes': 2,
            'class_names': str(['NB', 'NT']),
            'total_images': len(val_img_names),
            'evaluation_type': 'baseline',
        }
        mlflow.log_params(params_to_log)
        mlflow.log_input(ImageListDataset(val_img_names), context='baseline')

        # Compute baseline metrics
        val_map50, val_box_iou = compute_baseline_metrics(model, yolo_config_path, YOLO_DATASET_DIR, prefix="baseline")

        print(f"\n{'='*80}")
        print(f"BASELINE EVALUATION RESULTS:")
        print(f"  Model: {model_ckpt}")
        print(f"  Val Box IOU: {val_box_iou:.4f}")
        print(f"  Val mAP@50: {val_map50:.4f}")
        print(f"  MLflow Run ID: {mlflow_run_id}")
        print(f"{'='*80}\n")

        # Save results to JSON
        results = {
            'model_checkpoint': model_ckpt,
            'val_box_iou': val_box_iou,
            'val_map50': val_map50,
            'mlflow_run_id': mlflow_run_id,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(BASELINE_RUN_DIR / 'baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {BASELINE_RUN_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
