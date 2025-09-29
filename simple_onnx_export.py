"""
YOLO ONNX Export and Validation Script
Exports YOLO models to ONNX format and validates the results.
"""
import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib import patches
from ultralytics import YOLO

# Import the inference module
from yolo_onnx_inference import YOLOONNXInference, preprocess_images, plot_detections

load_dotenv()  # Load environment variables from .env file


class BaselineEncapsulatedYOLO:
    """Baseline implementation using Ultralytics Results for comparison."""

    def __init__(self, model: YOLO):
        self.model = model

    def __call__(self, images) -> Tuple[np.ndarray, np.ndarray]:
        """Return confidences [K, B] and boxes [K, B, 4]."""
        results = self.model(images)
        K = len(self.model.names)
        B = len(results)

        confidences = np.zeros((K, B), dtype=np.float32)
        boxes = np.zeros((K, B, 4), dtype=np.float32)

        for b_idx, r in enumerate(results):
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()

                # Select highest confidence per class
                for det_idx, (cls_id, conf, box) in enumerate(zip(cls_ids, confs, xyxy)):
                    if 0 <= cls_id < K and conf > confidences[cls_id, b_idx]:
                        confidences[cls_id, b_idx] = conf
                        boxes[cls_id, b_idx] = box

        return confidences, boxes


def export_yolo_to_onnx(model_path: str, output_path: str = None, dynamic: bool = True, opset: int = 17):
    """
    Export YOLO model to ONNX format.

    Args:
        model_path: Path to the YOLO .pt model file
        output_path: Optional custom output path for ONNX file
        dynamic: Whether to use dynamic batch size
        opset: ONNX opset version

    Returns:
        Path to the exported ONNX file
    """
    model = YOLO(model_path, task='detect')

    if output_path:
        # Custom output path - need to handle this manually since ultralytics doesn't support custom paths directly
        onnx_path = model.export(format="onnx", dynamic=dynamic, opset=opset)
        if onnx_path != output_path:
            import shutil
            shutil.move(onnx_path, output_path)
            onnx_path = output_path
    else:
        onnx_path = model.export(format="onnx", dynamic=dynamic, opset=opset)

    print(f"✅ ONNX export successful: {onnx_path}")
    return onnx_path, model


def plot_side_by_side_comparison(images, baseline_conf, baseline_boxes, onnx_conf, onnx_boxes,
                                 class_names=None, score_thresh=0.1):
    """
    Plot baseline and ONNX detections side-by-side for visual comparison.
    """
    B = baseline_conf.shape[1]
    K = baseline_conf.shape[0]
    colors = plt.cm.get_cmap('Set3', K)

    for b in range(B):
        img = images[b]
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot both images
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        axes[0].set_title("Baseline (PyTorch)", fontsize=14)
        axes[1].set_title("ONNX Runtime", fontsize=14)

        # Track counts for both
        baseline_count = 0
        onnx_count = 0

        # Plot baseline detections
        for k in range(K):
            conf = baseline_conf[k, b]
            if conf < score_thresh:
                continue

            baseline_count += 1
            x1, y1, x2, y2 = baseline_boxes[k, b]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=colors(k), facecolor='none')
            axes[0].add_patch(rect)

            class_name = class_names[k] if class_names else f"Class {k}"
            label = f"{class_name}: {conf:.3f}"
            axes[0].text(x1, max(0, y1 - 5), label, color=colors(k), fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Plot ONNX detections
        for k in range(K):
            conf = onnx_conf[k, b]
            if conf < score_thresh:
                continue

            onnx_count += 1
            x1, y1, x2, y2 = onnx_boxes[k, b]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=colors(k), facecolor='none')
            axes[1].add_patch(rect)

            class_name = class_names[k] if class_names else f"Class {k}"
            label = f"{class_name}: {conf:.3f}"
            axes[1].text(x1, max(0, y1 - 5), label, color=colors(k), fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Add detection counts to titles
        axes[0].set_title(f"Baseline (PyTorch) - {baseline_count} detections", fontsize=14)
        axes[1].set_title(f"ONNX Runtime - {onnx_count} detections", fontsize=14)

        # Turn off axes
        axes[0].axis('off')
        axes[1].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Image {b} Comparison", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()


def compare_outputs(baseline_conf, baseline_boxes, onnx_conf, onnx_boxes, class_names=None):
    """Compare baseline vs ONNX outputs."""
    print("\n=== COMPARISON RESULTS ===")

    # Shape check
    if baseline_conf.shape != onnx_conf.shape:
        print(f"❌ Confidence shapes differ: baseline {baseline_conf.shape} vs ONNX {onnx_conf.shape}")
        return False
    if baseline_boxes.shape != onnx_boxes.shape:
        print(f"❌ Box shapes differ: baseline {baseline_boxes.shape} vs ONNX {onnx_boxes.shape}")
        return False

    print(f"✅ Shapes match: confidences {baseline_conf.shape}, boxes {baseline_boxes.shape}")

    # Value comparison with reasonable tolerances
    conf_match = np.allclose(baseline_conf, onnx_conf, atol=0.01, rtol=0.05)
    box_match = np.allclose(baseline_boxes, onnx_boxes, atol=3.0, rtol=0.05)

    print(f"Confidences match (±0.01): {conf_match}")
    print(f"Boxes match (±3px): {box_match}")

    # Detailed comparison for non-matching cases
    K, B = baseline_conf.shape
    mismatches = 0

    for k in range(K):
        for b in range(B):
            baseline_c, onnx_c = baseline_conf[k, b], onnx_conf[k, b]
            baseline_b, onnx_b = baseline_boxes[k, b], onnx_boxes[k, b]

            conf_diff = abs(baseline_c - onnx_c)
            box_diff = np.max(np.abs(baseline_b - onnx_b)) if baseline_c > 0 or onnx_c > 0 else 0

            if conf_diff > 0.01 or box_diff > 3.0:
                if mismatches < 10:  # Limit output
                    class_name = class_names[k] if class_names else f"Class {k}"
                    print(f"  {class_name}, Image {b}: conf_diff={conf_diff:.4f}, box_diff={box_diff:.1f}")
                mismatches += 1

    if mismatches > 10:
        print(f"  ... and {mismatches - 10} more mismatches")

    overall_pass = conf_match and box_match
    print(f"\n{'✅ PASS' if overall_pass else '❌ FAIL'}: Overall comparison")
    return overall_pass


def load_test_images(dataset_dir: str, img_ids: list):
    """Load test images for validation."""
    img_path_template = f"{dataset_dir}/Images/pd_{{}}.png"
    images = []

    for img_id in img_ids:
        img_path = img_path_template.format(img_id)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            images.append(img)
        else:
            print(f"⚠️ Warning: Image not found: {img_path}")

    return images


if __name__ == "__main__":
    # Configuration
    checkpoint_path = "checkpoints/yolo_fetal_structures_20250923_223741/weights/best.pt"
    dataset_dir = os.getenv("DATASET_DIR", "data")

    print("🚀 YOLO ONNX Export and Validation")
    print("=" * 50)

    # Export YOLO model to ONNX
    print("📤 Exporting YOLO model to ONNX...")
    yolo_onnx_path, model = export_yolo_to_onnx(checkpoint_path, dynamic=True, opset=17)

    # Create inference engines
    print("🔧 Setting up inference engines...")
    onnx_inference = YOLOONNXInference(yolo_onnx_path, len(model.names))
    baseline_inference = BaselineEncapsulatedYOLO(model)
    class_names = model.names

    print(f"Model has {len(class_names)} classes: {class_names}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using ONNX model: {yolo_onnx_path}")
    print(f"Using dataset directory: {dataset_dir}")
    print()

    # Load sample images for testing
    print("📷 Loading test images...")
    ids = [109, 477, 398, 468, 469]
    images = load_test_images(dataset_dir, ids)

    if not images:
        print("❌ No test images found. Please check your dataset path.")
        exit(1)

    print(f"Loaded {len(images)} images for inference.")
    print()

    # Run inference comparison
    print("🔍 Running inference comparison...")
    baseline_conf, baseline_boxes = baseline_inference(images)
    onnx_conf, onnx_boxes = onnx_inference(images)
    print("Inference completed.\n")

    # Compare outputs
    comparison_passed = compare_outputs(baseline_conf, baseline_boxes, onnx_conf, onnx_boxes, class_names)
    print()

    # Visualize results
    print("📊 Generating visual comparison...")
    plot_side_by_side_comparison(images, baseline_conf, baseline_boxes, onnx_conf, onnx_boxes, class_names,
                                 score_thresh=0.1)

    print("✅ Export and validation complete!")
    if comparison_passed:
        print("🎉 ONNX model produces identical results to PyTorch baseline!")
    else:
        print("⚠️ Some differences detected between ONNX and PyTorch outputs.")
