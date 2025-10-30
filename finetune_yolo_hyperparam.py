import hashlib
import itertools
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import albumentations as A
import cv2
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
MLFLOW_EXPERIMENT_NAME = f"Fetal_Structures_yolo_{TARGET_HEIGHT}x{TARGET_WIDTH}_finetuned_solimed_hyperparam_search"
MLFLOW_USER = os.getenv("MLFLOW_USER")
MODEL_NAME = f"fetal_structures_yolo_{TARGET_HEIGHT}x{TARGET_WIDTH}_finetuned_solimed_{DATASET_VERSION}"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 2000
NUM_EXPERIMENTS = 100

# Hyperparameter Search Space Configuration
HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01],
    'weight_decay': [0.0001, 0.0005, 0.001],
    'batch_size': [16, 32],
    'early_stopping_patience': [200, 300, 400, 500, 1000],
    # Augmentation configurations (each is a preset)
    'augmentation_preset': ['lowest', 'light', 'medium', 'heavy', 'extreme', 'ultra']
}

# Define augmentation presets
AUGMENTATION_PRESETS = {
    'lowest': {
        'horizontal_flip_p': 0,
        'blur_p': 0.0,
        'noise_p': 0.1,
        'brightness_contrast_p': 0.1,
        'gamma_p': 0.1,
        'motion_blur_limit': 1,
        'gaussian_blur_limit': 1,
        'noise_std_range': (0.01, 0.02),
        'brightness_limit': 0.0,
        'contrast_limit': 0.0,
        'gamma_limit': (99, 100),
    },
    'light': {
        'horizontal_flip_p': 0.3,
        'blur_p': 0.1,
        'noise_p': 0.2,
        'brightness_contrast_p': 0.3,
        'gamma_p': 0.2,
        'motion_blur_limit': 3,
        'gaussian_blur_limit': 3,
        'noise_std_range': (0.03, 0.05),
        'brightness_limit': 0.1,
        'contrast_limit': 0.1,
        'gamma_limit': (95, 105),
    },
    'medium': {
        'horizontal_flip_p': 0.5,
        'blur_p': 0.3,
        'noise_p': 0.4,
        'brightness_contrast_p': 0.5,
        'gamma_p': 0.3,
        'motion_blur_limit': 5,
        'gaussian_blur_limit': 5,
        'noise_std_range': (0.05, 0.07),
        'brightness_limit': 0.15,
        'contrast_limit': 0.15,
        'gamma_limit': (90, 110),
    },
    'heavy': {
        'horizontal_flip_p': 0.7,
        'blur_p': 0.5,
        'noise_p': 0.5,
        'brightness_contrast_p': 0.6,
        'gamma_p': 0.4,
        'motion_blur_limit': 7,
        'gaussian_blur_limit': 7,
        'noise_std_range': (0.07, 0.10),
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'gamma_limit': (85, 115),
    },
    'extreme': {
        'horizontal_flip_p': 0.8,
        'blur_p': 0.7,
        'noise_p': 0.6,
        'brightness_contrast_p': 0.7,
        'gamma_p': 0.5,
        'motion_blur_limit': 9,
        'gaussian_blur_limit': 9,
        'noise_std_range': (0.08, 0.12),
        'brightness_limit': 0.25,
        'contrast_limit': 0.25,
        'gamma_limit': (80, 120),
    },
    'ultra': {
        'horizontal_flip_p': 0.9,
        'blur_p': 0.8,
        'noise_p': 0.7,
        'brightness_contrast_p': 0.8,
        'gamma_p': 0.6,
        'motion_blur_limit': 11,
        'gaussian_blur_limit': 11,
        'noise_std_range': (0.10, 0.15),
        'brightness_limit': 0.3,
        'contrast_limit': 0.3,
        'gamma_limit': (75, 125),
    }
}


@dataclass
class HyperparamConfig:
    """Dataclass to store hyperparameter configuration"""
    learning_rate: float
    weight_decay: float
    batch_size: int
    early_stopping_patience: int
    augmentation_preset: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_aug_params(self) -> Dict[str, Any]:
        return AUGMENTATION_PRESETS[self.augmentation_preset]


@dataclass
class ExperimentResult:
    """Store results from an experiment"""
    config: HyperparamConfig
    val_box_iou: float
    val_map50: float
    mlflow_run_id: str
    best_model_path: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'val_box_iou': self.val_box_iou,
            'val_map50': self.val_map50,
            'mlflow_run_id': self.mlflow_run_id,
            'best_model_path': self.best_model_path,
            'timestamp': self.timestamp
        }


class HyperparamSearchTracker:
    """Track all experiments and best results"""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.best_iou_result: Optional[ExperimentResult] = None
        self.best_map50_result: Optional[ExperimentResult] = None

    def add_result(self, result: ExperimentResult):
        self.results.append(result)

        # Update best IOU
        if self.best_iou_result is None or result.val_box_iou > self.best_iou_result.val_box_iou:
            self.best_iou_result = result

        # Update best mAP@50
        if self.best_map50_result is None or result.val_map50 > self.best_map50_result.val_map50:
            self.best_map50_result = result

        self._save_results()

    def _save_results(self):
        """Save all results to JSON"""
        data = {
            'all_results': [r.to_dict() for r in self.results],
            'best_iou': self.best_iou_result.to_dict() if self.best_iou_result else None,
            'best_map50': self.best_map50_result.to_dict() if self.best_map50_result else None,
        }

        with open(self.save_dir / 'hyperparam_search_results.json', 'w') as f:
            json.dump(data, f, indent=2)

        # Also save a summary CSV
        df = pd.DataFrame([
            {
                **r.config.to_dict(),
                'val_box_iou': r.val_box_iou,
                'val_map50': r.val_map50,
                'mlflow_run_id': r.mlflow_run_id,
                'timestamp': r.timestamp
            }
            for r in self.results
        ])
        df.to_csv(self.save_dir / 'hyperparam_search_summary.csv', index=False)

    def print_summary(self):
        """Print summary of best results"""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 80)
        print(f"\nTotal experiments run: {len(self.results)}")

        if self.best_iou_result:
            print("\n" + "-" * 80)
            print("BEST VAL BOX IOU:")
            print(f"  IOU: {self.best_iou_result.val_box_iou:.4f}")
            print(f"  mAP@50: {self.best_iou_result.val_map50:.4f}")
            print(f"  Config: {json.dumps(self.best_iou_result.config.to_dict(), indent=4)}")
            print(f"  MLflow Run ID: {self.best_iou_result.mlflow_run_id}")
            print(f"  Model Path: {self.best_iou_result.best_model_path}")

        if self.best_map50_result:
            print("\n" + "-" * 80)
            print("BEST VAL mAP@50:")
            print(f"  mAP@50: {self.best_map50_result.val_map50:.4f}")
            print(f"  IOU: {self.best_map50_result.val_box_iou:.4f}")
            print(f"  Config: {json.dumps(self.best_map50_result.config.to_dict(), indent=4)}")
            print(f"  MLflow Run ID: {self.best_map50_result.mlflow_run_id}")
            print(f"  Model Path: {self.best_map50_result.best_model_path}")
        print("=" * 80 + "\n")


def generate_hyperparam_configs(search_space: Dict[str, List],
                                max_configs: int = None,
                                random_sample: bool = False) -> List[HyperparamConfig]:
    """Generate hyperparameter configurations from search space"""

    if random_sample and max_configs:
        # Random sampling
        configs = []
        keys = list(search_space.keys())
        for _ in range(max_configs):
            config_dict = {}
            for k in keys:
                val = np.random.choice(search_space[k])
                # Convert numpy types to native Python types
                if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    val = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                    val = float(val)
                config_dict[k] = val
            configs.append(HyperparamConfig(**config_dict))
    else:
        # Grid search
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        all_combinations = list(itertools.product(*values))

        if max_configs and len(all_combinations) > max_configs:
            # Randomly sample if too many combinations
            indices = np.random.choice(len(all_combinations), max_configs, replace=False)
            all_combinations = [all_combinations[i] for i in indices]

        configs = [HyperparamConfig(**dict(zip(keys, combo))) for combo in all_combinations]

    return configs


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


def create_train_val_test_split(annotations_df, n_val_videos=None, n_test_videos=None):
    """Create train/val/test split. Test is same size as val by default."""
    unique_img_paths = annotations_df['file_path'].unique()
    np.random.shuffle(unique_img_paths)

    # Calculate splits
    if n_test_videos is not None:
        n_test = min(n_test_videos, len(unique_img_paths) // 3)
    else:
        n_test = int(len(unique_img_paths) * 0.15)

    if n_val_videos is not None:
        n_val = min(n_val_videos, len(unique_img_paths) - n_test)
    else:
        n_val = n_test  # Same size as test

    # Split in order: test, val, train (so test is never touched)
    test_img_names = unique_img_paths[:n_test].tolist()
    val_img_names = unique_img_paths[n_test:n_test + n_val].tolist()
    train_img_names = unique_img_paths[n_test + n_val:].tolist()

    print(f"Total images: {len(unique_img_paths)}")
    print(f"Train: {len(train_img_names)}, Val: {len(val_img_names)}, Test: {len(test_img_names)}")
    print(
        f"Split: {len(train_img_names) / len(unique_img_paths):.1%} / {len(val_img_names) / len(unique_img_paths):.1%} / {len(test_img_names) / len(unique_img_paths):.1%}")

    # Create annotations dict for all images
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

    return train_img_names, val_img_names, test_img_names, ANNOTATIONS


def get_transforms(aug_params: Dict[str, Any] = None):
    """Define data augmentation transforms with configurable parameters."""
    if aug_params is None:
        # Default medium augmentation
        aug_params = AUGMENTATION_PRESETS['medium']

    train_transform = A.Compose([
        A.HorizontalFlip(p=aug_params['horizontal_flip_p']),
        A.SmallestMaxSize(max_size=int(TARGET_HEIGHT * 1.01), p=1),
        A.RandomCrop(height=TARGET_HEIGHT, width=TARGET_WIDTH, p=1),
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH, p=1),
        A.OneOf([
            A.MotionBlur(blur_limit=aug_params['motion_blur_limit'], p=1.0),
            A.GaussianBlur(blur_limit=aug_params['gaussian_blur_limit'], p=1.0),
        ], p=aug_params['blur_p']),
        A.OneOf([
            A.GaussNoise(std_range=aug_params['noise_std_range'], p=1.0),
            A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=1.0),
        ], p=aug_params['noise_p']),
        A.RandomBrightnessContrast(
            brightness_limit=aug_params['brightness_limit'],
            contrast_limit=aug_params['contrast_limit'],
            p=aug_params['brightness_contrast_p']
        ),
        A.RandomGamma(gamma_limit=aug_params['gamma_limit'], p=aug_params['gamma_p']),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})

    val_transform = A.Compose([
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})

    return train_transform, val_transform


class FinetuneDetectionDataset(Dataset):
    def __init__(self, image_root: str, img_names: list[str],
                 annotations: dict[str, list[tuple[int, tuple[int, int, int, int]]]],
                 transform: A.Compose | None = None):
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
        super().__init__()
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
        'train': 'train/images',
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


def mlflow_callback(trainer):
    epoch = trainer.epoch
    for k, v in trainer.metrics.items():
        if isinstance(v, (int, float)):
            safe_k = str(k).replace('(', '_').replace(')', '').replace('[', '_').replace(']', '').replace(' ', '_')
            mlflow.log_metric(safe_k, float(v), step=epoch)


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {MLFLOW_URI}")
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")


def run_single_experiment(config: HyperparamConfig,
                          train_img_names: list,
                          val_img_names: list,
                          ANNOTATIONS: dict,
                          experiment_idx: int,
                          total_experiments: int) -> Tuple[ExperimentResult, str]:
    """Run a single training experiment with given hyperparameter configuration."""

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT {experiment_idx + 1}/{total_experiments}")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    print(f"{'=' * 80}\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Train: {len(train_img_names)}, Val: {len(val_img_names)}")

    # Get transforms with augmentation parameters
    aug_params = config.get_aug_params()
    train_transform, val_transform = get_transforms(aug_params)

    # Create datasets
    train_dataset = FinetuneDetectionDataset(
        image_root=DATASET_IMAGES_DIR,
        img_names=train_img_names,
        annotations=ANNOTATIONS,
        transform=train_transform
    )
    val_dataset = FinetuneDetectionDataset(
        image_root=DATASET_IMAGES_DIR,
        img_names=val_img_names,
        annotations=ANNOTATIONS,
        transform=val_transform
    )

    # Prepare YOLO dataset
    FINETUNE_RUN_DIR = Path('runs') / 'hyperparam_search' / f'{timestamp}_exp_{experiment_idx + 1}'
    YOLO_DATASET_DIR = FINETUNE_RUN_DIR / 'dataset'
    YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    train_prep = YOLODatasetPreparator(train_dataset, YOLO_DATASET_DIR)
    val_prep = YOLODatasetPreparator(val_dataset, YOLO_DATASET_DIR)
    train_prep.prepare_split('train')
    val_prep.prepare_split('val')

    yolo_config_path = create_yolo_config(YOLO_DATASET_DIR, train_dataset)

    # Train model
    model_ckpt = BEST_MODEL_PATH if BEST_MODEL_PATH and os.path.exists(BEST_MODEL_PATH) else 'yolov8n.pt'
    print(f'Loading model checkpoint: {model_ckpt}')
    model = YOLO(model_ckpt)

    train_args = {
        'data': str(yolo_config_path),
        'epochs': NUM_EPOCHS,
        'imgsz': max(TARGET_HEIGHT, TARGET_WIDTH),
        'batch': config.batch_size,
        'lr0': config.learning_rate,
        'weight_decay': config.weight_decay,
        'patience': config.early_stopping_patience,
        'project': str(FINETUNE_RUN_DIR.parent),
        'name': FINETUNE_RUN_DIR.name,
        'exist_ok': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'device': DEVICE,
        'workers': 0,
    }

    # Helper function to convert numpy types to native Python types
    def convert_to_native_types(obj):
        """Convert numpy/pandas types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        return obj

    # Log to MLflow
    with mlflow.start_run(run_name=f"{MLFLOW_USER}_exp_{experiment_idx + 1}_{timestamp}") as run:
        mlflow_run_id = run.info.run_id

        # Convert augmentation parameters to native types
        aug_params_native = convert_to_native_types(aug_params)

        # Log hyperparameters
        params_to_log = {
            'model_type': 'YOLOv8',
            'finetune_from': model_ckpt,
            'num_classes': 2,
            'class_names': str(['NB', 'NT']),
            'train_images': len(train_img_names),
            'val_images': len(val_img_names),
            **config.to_dict(),
            **{f'aug_{k}': str(v) if isinstance(v, (tuple, list)) else v for k, v in aug_params_native.items()},
        }
        # Add train_args separately
        for k, v in train_args.items():
            params_to_log[f'train_{k}'] = str(v) if isinstance(v, (Path, tuple, list)) else v

        # Convert all params to native types
        params_to_log = convert_to_native_types(params_to_log)

        mlflow.log_params(params_to_log)
        mlflow.log_input(ImageListDataset(train_img_names), context='train')
        mlflow.log_input(ImageListDataset(val_img_names), context='val')

        # Train
        model.add_callback("on_fit_epoch_end", mlflow_callback)
        results = model.train(**train_args)

        # Log training results
        if hasattr(results, 'results_dict'):
            for k, v in results.results_dict.items():
                if isinstance(v, (int, float)):
                    safe_k = str(k).replace('(', '_').replace(')', '').replace('[', '_').replace(']', '').replace(' ',
                                                                                                                  '_')
                    mlflow.log_metric(f"train/{safe_k}", float(v))

        # Log best model
        best_pt = FINETUNE_RUN_DIR / 'weights' / 'best.pt'
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt))

        # Log per-epoch metrics
        results_csv = FINETUNE_RUN_DIR / 'results.csv'
        if results_csv.exists():
            try:
                df_results = pd.read_csv(results_csv)
                exclude_cols = {'epoch', 'time'}
                exclude_prefixes = ('lr/',)
                for _, row in df_results.iterrows():
                    step = int(row['epoch']) if 'epoch' in row and not pd.isna(row['epoch']) else None
                    for col in df_results.columns:
                        if col in exclude_cols or any(col.startswith(p) for p in exclude_prefixes):
                            continue
                        val = row[col]
                        if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                            safe_col = str(col).replace('(', '_').replace(')', '').replace('[', '_').replace(']',
                                                                                                             '').replace(
                                ' ', '_')
                            if step is not None:
                                mlflow.log_metric(f"train_epoch/{safe_col}", float(val), step=step)
                            else:
                                mlflow.log_metric(f"train_epoch/{safe_col}", float(val))
            except Exception as e:
                print(f"Warning: failed to log per-epoch metrics: {e}")

        # Validation
        val_results = model.val(data=str(yolo_config_path), split='val', verbose=True, plots=True)

        # Extract validation metrics
        val_map50 = 0.0
        if hasattr(val_results, 'results_dict'):
            for k, v in val_results.results_dict.items():
                if isinstance(v, (int, float)):
                    safe_k = str(k).replace('(', '_').replace(')', '').replace('[', '_').replace(']', '').replace(' ',
                                                                                                                  '_')
                    mlflow.log_metric(f"val/{safe_k}", float(v))
                    if 'map50' in str(k).lower() or 'mAP50' in str(k):
                        val_map50 = float(v)

        # Custom IOU evaluation
        all_ious = []
        val_images_dir = YOLO_DATASET_DIR / 'val' / 'images'
        for img_file in list(val_images_dir.glob('*.jpg'))[:500]:
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
        mlflow.log_metric('val_box_iou_mean', val_box_iou)

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {experiment_idx + 1} RESULTS:")
        print(f"  Val Box IOU: {val_box_iou:.4f}")
        print(f"  Val mAP@50: {val_map50:.4f}")
        print(f"{'=' * 80}\n")

        # Create result object
        result = ExperimentResult(
            config=config,
            val_box_iou=val_box_iou,
            val_map50=val_map50,
            mlflow_run_id=mlflow_run_id,
            best_model_path=str(best_pt) if best_pt.exists() else "",
            timestamp=timestamp
        )

        return result, mlflow_run_id


def evaluate_best_model_on_test(best_result: ExperimentResult,
                                test_img_names: list,
                                ANNOTATIONS: dict,
                                experiment_name: str) -> Dict[str, float]:
    """Evaluate the best model on the test set - works exactly like validation."""

    print(f"\n{'=' * 80}")
    print("EVALUATING BEST MODEL ON TEST SET")
    print(f"{'=' * 80}")
    print(f"Best model: {best_result.best_model_path}")
    print(f"Best val IOU: {best_result.val_box_iou:.4f}")
    print(f"Best val mAP@50: {best_result.val_map50:.4f}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(best_result.best_model_path):
        print(f"Error: Model not found at {best_result.best_model_path}")
        return {}

    model = YOLO(best_result.best_model_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create test dataset (no augmentation - same as val)
    aug_params = best_result.config.get_aug_params()
    _, test_transform = get_transforms(aug_params)

    test_dataset = FinetuneDetectionDataset(
        image_root=DATASET_IMAGES_DIR,
        img_names=test_img_names,
        annotations=ANNOTATIONS,
        transform=test_transform
    )

    # Prepare test split
    TEST_RUN_DIR = Path('runs') / 'hyperparam_search' / f'{experiment_name}_test_{timestamp}'
    YOLO_TEST_DIR = TEST_RUN_DIR / 'dataset'
    YOLO_TEST_DIR.mkdir(parents=True, exist_ok=True)

    test_prep = YOLODatasetPreparator(test_dataset, YOLO_TEST_DIR)
    test_prep.prepare_split('val')  # Use 'val' name for YOLO compatibility

    test_config_path = create_yolo_config(YOLO_TEST_DIR, test_dataset)

    # Run validation on test set
    print("\nRunning evaluation on test set...")
    test_results = model.val(data=str(test_config_path), split='val', verbose=True, plots=True)

    # Extract metrics
    test_metrics = {}
    test_map50 = 0.0
    if hasattr(test_results, 'results_dict'):
        for k, v in test_results.results_dict.items():
            if isinstance(v, (int, float)):
                safe_k = str(k).replace('(', '_').replace(')', '').replace('[', '_').replace(']', '').replace(' ', '_')
                test_metrics[f"test/{safe_k}"] = float(v)
                if 'map50' in str(k).lower() or 'mAP50' in str(k):
                    test_map50 = float(v)

    # Custom IOU calculation (same as val)
    print("\nCalculating custom box IOU on test set...")
    all_ious = []
    test_images_dir = YOLO_TEST_DIR / 'val' / 'images'

    for img_file in list(test_images_dir.glob('*.jpg')):
        label_file = test_images_dir.parent / 'labels' / (img_file.stem + '.txt')
        gt_by_class = {}
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(float(parts[0]))
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
        if classes:
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

    test_box_iou = float(np.mean(all_ious)) if all_ious else 0.0
    test_metrics['test_box_iou_mean'] = test_box_iou
    test_metrics['test_map50'] = test_map50

    # Log to MLflow
    with mlflow.start_run(run_id=best_result.mlflow_run_id):
        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_param('test_evaluated', True)
        mlflow.log_param('test_images', len(test_img_names))

    print(f"\n{'=' * 80}")
    print("TEST SET RESULTS:")
    print(f"  Test Box IOU: {test_box_iou:.4f}")
    print(f"  Test mAP@50: {test_map50:.4f}")
    print(f"  Test images: {len(test_img_names)}")
    print(f"{'=' * 80}\n")

    return test_metrics


def main():
    """Main function to run hyperparameter search"""

    # Setup
    setup_mlflow()
    annotations_df = load_and_preprocess_data()

    # Create fixed train/val/test split once (70%/15%/15%)
    print(f"\n{'=' * 80}")
    print("CREATING FIXED TRAIN/VAL/TEST SPLIT")
    print(f"{'=' * 80}\n")

    unique_img_paths = annotations_df['file_path'].unique()
    np.random.shuffle(unique_img_paths)

    total_images = len(unique_img_paths)
    n_train = int(total_images * 0.7)
    n_val = int(total_images * 0.15)
    n_test = total_images - n_train - n_val

    train_img_names = unique_img_paths[:n_train].tolist()
    val_img_names = unique_img_paths[n_train:n_train + n_val].tolist()
    test_img_names = unique_img_paths[n_train + n_val:].tolist()

    # Create full annotations dict (includes all splits)
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

    print(f"Total images: {total_images}")
    print(f"Train: {len(train_img_names)} ({len(train_img_names) / total_images:.1%})")
    print(f"Val: {len(val_img_names)} ({len(val_img_names) / total_images:.1%})")
    print(f"Test: {len(test_img_names)} ({len(test_img_names) / total_images:.1%})")
    print(f"{'=' * 80}\n")

    # Generate hyperparameter configurations
    configs = generate_hyperparam_configs(
        HYPERPARAM_SEARCH_SPACE,
        max_configs=NUM_EXPERIMENTS,
        random_sample=True
    )

    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER SEARCH")
    print(f"{'=' * 80}")
    print(f"Total configurations to test: {len(configs)}")
    print(f"{'=' * 80}\n")

    # Initialize tracker
    tracker_dir = Path('runs') / 'hyperparam_search' / datetime.now().strftime('%Y%m%d_%H%M%S')
    tracker = HyperparamSearchTracker(tracker_dir)
    experiment_name = tracker_dir.name

    # Run experiments on train/val only (test set held out)
    for idx, config in enumerate(configs):
        try:
            result, run_id = run_single_experiment(
                config,
                train_img_names,
                val_img_names,
                ANNOTATIONS,
                idx,
                len(configs)
            )
            tracker.add_result(result)

            # Print progress
            print(f"\n{'=' * 80}")
            print(f"PROGRESS: {idx + 1}/{len(configs)} experiments completed")
            if tracker.best_iou_result:
                print(f"Current Best IOU: {tracker.best_iou_result.val_box_iou:.4f}")
            if tracker.best_map50_result:
                print(f"Current Best mAP@50: {tracker.best_map50_result.val_map50:.4f}")
            print(f"{'=' * 80}\n")

        except Exception as e:
            print(f"\n{'!' * 80}")
            print(f"ERROR in experiment {idx + 1}: {str(e)}")
            print(f"{'!' * 80}\n")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    tracker.print_summary()

    # Evaluate best model on test set
    if tracker.best_iou_result:
        try:
            test_metrics = evaluate_best_model_on_test(
                tracker.best_iou_result,
                test_img_names,
                ANNOTATIONS,
                experiment_name
            )

            # Save test results
            test_results_path = tracker.save_dir / 'test_set_results.json'
            with open(test_results_path, 'w') as f:
                json.dump({
                    'test_metrics': test_metrics,
                    'best_config': tracker.best_iou_result.config.to_dict(),
                    'val_box_iou': tracker.best_iou_result.val_box_iou,
                    'val_map50': tracker.best_iou_result.val_map50,
                    'test_images': len(test_img_names)
                }, f, indent=2)

            print(f"\n✅ Test results saved to: {test_results_path}")

        except Exception as e:
            print(f"\n❌ Error evaluating on test set: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\nAll results saved to: {tracker.save_dir}")
    print(f"  - {tracker.save_dir / 'hyperparam_search_results.json'}")
    print(f"  - {tracker.save_dir / 'hyperparam_search_summary.csv'}")
    print(f"  - {tracker.save_dir / 'test_set_results.json'}")


if __name__ == "__main__":
    main()
