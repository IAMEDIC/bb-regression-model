#!/usr/bin/env python3
"""
Compute test metrics for each finetune YOLO run

This script builds a 'test' split per run as the complement of the images used in that run's train+val (relative to the base dataset). It then evaluates `weights/best.pt` on that constructed test split using Ultralytics' `model.val(split='test')`.

Notes:
- Base dataset assumed at DATASET_IMAGES_DIR (from FINETUNE_DIR env var + "media") with flat structure of images.
- Per-run datasets are at `runs/hyperparam_search/<run_id>/dataset` and contain train/ and val/.
- For evaluation, we create a per-run folder `test_eval/<timestamp>/dataset` with a dataset.yaml that points to existing train/val and a constructed test/ containing symlinks to the complement images and labels created on the fly.
- Results are saved per run as `test_eval/<timestamp>/test_metrics.json` and aggregated to a CSV.
"""

from __future__ import annotations
from pathlib import Path
import os, json, csv, time, shutil
from typing import Dict, List, Tuple, Iterable, Optional
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
# Config
REPO_ROOT = Path('.').resolve()
RUNS_ROOT = REPO_ROOT / 'runs' / 'hyperparam_search'
FINETUNE_DS_DIR = os.getenv("FINETUNE_DIR")
DATASET_IMAGES_DIR_NAME = "media"
DATASET_IMAGES_DIR = os.path.join(FINETUNE_DS_DIR, DATASET_IMAGES_DIR_NAME)
BASE_DATASET = Path(DATASET_IMAGES_DIR)
RUN_GLOB = '*'  # change to a different pattern (e.g., '*20251024*_exp_*') or '*' to process all
MAX_TEST_IMAGES: Optional[int] = None  # set to an int for a faster dry-run; None for full set
SMOKE_TEST: bool = False  # set True to run a tiny test on a single run
DEVICE: Optional[str] = None  # e.g., '0' for GPU 0, or 'cpu'

assert BASE_DATASET.exists(), f'Base dataset not found: {BASE_DATASET}'
print('Repo root:', REPO_ROOT)
print('Runs root:', RUNS_ROOT)
print('Base dataset:', BASE_DATASET)

# Load annotations and metadata
ANNOTATIONS_PATH = os.path.join(FINETUNE_DS_DIR, "picture_bb_annotations.csv")
IMG_METADATA_PATH = os.path.join(FINETUNE_DS_DIR, "media.csv")
annotations_df = pd.read_csv(ANNOTATIONS_PATH)
img_metadata_df = pd.read_csv(IMG_METADATA_PATH)

# Preprocess annotations
annotations_df = annotations_df[annotations_df['bb_class'].isin(['NB', 'NT'])]
annotations_df = annotations_df[annotations_df['media_type'] == 'frame']
NUMBER_OF_CLASSES = len(pd.unique(annotations_df['bb_class']))
# Merge with media metadata to get file paths
annotations_df = annotations_df.merge(
    img_metadata_df[['id', 'file_path', 'filename']],
    left_on='media_id',
    right_on='id',
    suffixes=('', '_media')
)
# Convert x_min, y_min, width, height to integers
annotations_df['x_min'] = annotations_df['x_min'].astype(int)
annotations_df['y_min'] = annotations_df['y_min'].astype(int)

# Create ANNOTATIONS dict
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



# Utilities to index base dataset and per-run datasets
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS]

def index_base_dataset(base_root: Path) -> Dict[str, Path]:
    img2path: Dict[str, Path] = {}
    for img in list_images(base_root):
        key = img.name
        img2path[key] = img
    return img2path

def get_run_train_val_images(run_dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    train_imgs = list_images(run_dataset_root / 'train' / 'images')
    val_imgs = list_images(run_dataset_root / 'val' / 'images')
    return train_imgs, val_imgs

def complement_test_selection(train_imgs: Iterable[Path], val_imgs: Iterable[Path],
                               base_img_index: Dict[str, Path],
                               max_count: Optional[int] = None) -> List[Path]:
    used = {p.name for p in train_imgs} | {p.name for p in val_imgs}
    selected: List[Path] = []
    for key, img_path in base_img_index.items():
        if key in used:
            continue
        selected.append(img_path)
        if max_count is not None and len(selected) >= max_count:
            break
    return selected

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass

def build_eval_dataset(run_dir: Path, run_dataset: Path, names: List[str], nc: int,
                       test_imgs: List[Path], tag: Optional[str] = None) -> Tuple[Path, Path]:
    ts = time.strftime('%Y%m%d_%H%M%S')
    tag = tag or ts
    eval_root = run_dir / 'test_eval' / tag
    ds_root = eval_root / 'dataset'
    # Create test split with symlinks
    test_img_dir = ds_root / 'test' / 'images'
    test_lbl_dir = ds_root / 'test' / 'labels'
    for img_path in test_imgs:
        safe_symlink(img_path, test_img_dir / img_path.name)
        lbl_dst = test_lbl_dir / (img_path.stem + '.txt')
    # Build dataset.yaml referencing absolute paths for train/val/test
    
    yaml = {
        'path': str(ds_root),
        'train': str((run_dataset / 'train' / 'images').resolve()),
        'val': str((run_dataset / 'val' / 'images').resolve()),
        'test': str(test_img_dir.resolve()),
        'names': names,
        'nc': int(nc),
    }
    ypath = ds_root / 'dataset.yaml'
    ypath.parent.mkdir(parents=True, exist_ok=True)
    ypath.write_text(json.dumps(yaml, indent=2).replace('\\n', '\n'), encoding='utf-8')
    return eval_root, ypath

def read_run_dataset_yaml(run_dataset_root: Path) -> Tuple[List[str], int]:
    import yaml as pyyaml
    ypath = run_dataset_root / 'dataset.yaml'
    data = pyyaml.safe_load(ypath.read_text(encoding='utf-8'))
    names = data.get('names')
    nc = data.get('nc')
    assert isinstance(names, list) and isinstance(nc, int), f'Invalid dataset.yaml at {ypath}'
    return names, nc

def evaluate_test_with_yolo(best_pt: Path, data_yaml: Path, device: Optional[str] = None) -> Dict:
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    kwargs = {}
    if device:
        kwargs['device'] = device
    results = model.val(data=str(data_yaml), split='test', verbose=True, **kwargs)
    metrics = {}
    try:
        metrics = getattr(results, 'results_dict', None) or {}
    except Exception:
        metrics = {}
    for k in ('box', 'boxes', 'metrics'):
        v = getattr(results, k, None)
        if isinstance(v, dict):
            metrics.update(v)
    if 'mAP50_B' in metrics:
        metrics['test_map50'] = metrics.get('mAP50_B')
    if 'box/mAP50' in metrics:
        metrics['test_map50'] = metrics.get('box/mAP50')
    return metrics

def discover_runs(runs_root: Path, pattern: str) -> List[Path]:
    runs = sorted([p for p in runs_root.glob(pattern) if p.is_dir()])
    filtered = []
    for r in runs:
        if (r / 'dataset').exists() and (r / 'weights' / 'best.pt').exists():
            filtered.append(r)
    return filtered

if __name__ == "__main__":
    # Index base dataset
    img2path = index_base_dataset(BASE_DATASET)
    print('Base images indexed:', len(img2path))

    # Discover runs to process
    run_dirs = discover_runs(RUNS_ROOT, RUN_GLOB)
    print('Found runs:', len(run_dirs))
    if run_dirs:
        print('First 3 runs:', [p.name for p in run_dirs[:3]])

    # Load existing results
    results_file = RUNS_ROOT / '20251029_105157' / 'hyperparam_search_results.json'
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    all_results = results_data['all_results']

    summary_csv = RUNS_ROOT / '20251029_105157' / 'hyperparam_search_summary.csv'
    df = pd.read_csv(summary_csv)
    if 'test_map50' not in df.columns:
        df['test_map50'] = 0.0
    if 'test_box_iou' not in df.columns:
        df['test_box_iou'] = 0.0

    # Main loop: build test split, evaluate, update existing results
    for run_dir in run_dirs:
        timestamp = run_dir.name.split('_exp_')[0]
        result = next((r for r in all_results if r['timestamp'] == timestamp), None)
        if not result:
            print(f"No result found for timestamp {timestamp}, skipping")
            continue

        run_dataset = run_dir / 'dataset'
        best_pt = run_dir / 'weights' / 'best.pt'
        names, nc = read_run_dataset_yaml(run_dataset)
        train_imgs, val_imgs = get_run_train_val_images(run_dataset)
        test_imgs = complement_test_selection(train_imgs, val_imgs, img2path, MAX_TEST_IMAGES)
        print(f"Run {run_dir.name}: train={len(train_imgs)}, val={len(val_imgs)}, test_sel={len(test_imgs)}")
        eval_root, data_yaml = build_eval_dataset(run_dir, run_dataset, names, nc, test_imgs)
        metrics = evaluate_test_with_yolo(best_pt, data_yaml, DEVICE)
        # Save per-run metrics
        out_json = eval_root / 'test_metrics.json'
        out_json.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        # Update results
        result['test_map50'] = metrics.get('test_map50', 0)
        result['test_box_iou'] = 0.0  # Placeholder, compute if needed
        df.loc[df['timestamp'] == timestamp, 'test_map50'] = metrics.get('test_map50', 0)
        df.loc[df['timestamp'] == timestamp, 'test_box_iou'] = 0.0

    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    df.to_csv(summary_csv, index=False)
    print('Updated hyperparam_search_results.json and hyperparam_search_summary.csv')

    # Optional smoke test
    if SMOKE_TEST:
        test_run = run_dirs[0] if run_dirs else None
        if test_run:
            MAX_TEST_IMAGES = 2
            run_dataset = test_run / 'dataset'
            names, nc = read_run_dataset_yaml(run_dataset)
            train_imgs, val_imgs = get_run_train_val_images(run_dataset)
            test_imgs = complement_test_selection(train_imgs, val_imgs, img2path, MAX_TEST_IMAGES)
            eval_root, data_yaml = build_eval_dataset(test_run, run_dataset, names, nc, test_imgs, tag='smoke')
            metrics = evaluate_test_with_yolo(test_run / 'weights' / 'best.pt', data_yaml, DEVICE)
            print('Smoke test metrics:', metrics)
        else:
            print('No runs found for smoke test')
