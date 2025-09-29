"""
Simplified ONNX export that works around the raw output complexity.
Two-stage inference: YOLO ONNX (raw) + Python postprocessing + Aggregation
"""
import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from matplotlib import patches
from ultralytics import YOLO

load_dotenv()  # Load environment variables from .env file2


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Letterbox image for YOLO inference."""
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h, new_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - nw, new_h - nh
    dw, dh = pad_w // 2, pad_h // 2
    img_padded = cv2.copyMakeBorder(img_resized, dh, pad_h - dh, dw, pad_w - dw, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, dw, dh


def preprocess_images(images, imgsz=(640, 640)):
    """Preprocess images for ONNX inference."""
    batch = []
    metas = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lb, r, dw, dh = letterbox(img_rgb, new_shape=imgsz)
        x = lb.astype('float32') / 255.0
        x = x.transpose(2, 0, 1)  # HWC -> CHW
        batch.append(x)
        metas.append({"r": r, "dw": float(dw), "dh": float(dh), "orig_shape": img.shape[:2]})
    return np.stack(batch, axis=0), metas


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


class TwoStageONNXInference:
    """Pure ONNX inference: raw YOLO ONNX + pure NumPy postprocessing."""

    def __init__(self, yolo_onnx_path: str, num_classes: int):
        self.sess = ort.InferenceSession(yolo_onnx_path, providers=["CPUExecutionProvider"])
        self.num_classes = num_classes

        # Check ONNX input shape to see if it supports batching
        input_shape = self.sess.get_inputs()[0].shape
        batch_dim = input_shape[0]
        if isinstance(batch_dim, str) or batch_dim == -1 or (isinstance(batch_dim, int) and batch_dim > 1):
            self.supports_batching = True
        else:
            self.supports_batching = False
        print(f"ONNX model input shape: {input_shape}, supports batching: {self.supports_batching}")

    def _postprocess_onnx_output(self, raw_output, conf_threshold=0.25, nms_threshold=0.45):
        """
        Convert raw ONNX output [B, 6, 8400] to filtered detections.
        Returns: List of detection arrays, one per batch item.
        Each detection array has shape [N, 6] = [x1, y1, x2, y2, conf, class_id]
        """
        batch_size = raw_output.shape[0]
        detections_per_batch = []

        for b in range(batch_size):
            # Extract detections for this batch item: [6, 8400] -> [8400, 6]
            dets = raw_output[b].T  # [8400, 6]

            # Extract components
            x_center, y_center, width, height = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
            confidence = dets[:, 4]
            class_id = dets[:, 5].astype(int)

            # Filter by confidence
            valid_mask = confidence > conf_threshold
            if not np.any(valid_mask):
                detections_per_batch.append(np.empty((0, 6)))
                continue

            # Apply mask
            x_center = x_center[valid_mask]
            y_center = y_center[valid_mask]
            width = width[valid_mask]
            height = height[valid_mask]
            confidence = confidence[valid_mask]
            class_id = class_id[valid_mask]

            # Convert center format to xyxy format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Stack into final format [N, 6]
            batch_detections = np.column_stack([x1, y1, x2, y2, confidence, class_id])

            # Simple NMS per class (optional, can be disabled for our use case)
            if nms_threshold < 1.0:
                batch_detections = self._apply_nms_per_class(batch_detections, nms_threshold)

            detections_per_batch.append(batch_detections)

        return detections_per_batch

    def _apply_nms_per_class(self, detections, nms_threshold):
        """Apply NMS per class to detections [N, 6]."""
        if len(detections) == 0:
            return detections

        kept_detections = []
        unique_classes = np.unique(detections[:, 5])

        for cls in unique_classes:
            cls_mask = detections[:, 5] == cls
            cls_dets = detections[cls_mask]

            # Simple NMS implementation
            indices = self._nms(cls_dets[:, :4], cls_dets[:, 4], nms_threshold)
            kept_detections.append(cls_dets[indices])

        if kept_detections:
            return np.vstack(kept_detections)
        else:
            return np.empty((0, 6))

    def _nms(self, boxes, scores, threshold):
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return []

        # Calculate areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union

            # Keep boxes with IoU below threshold
            order = order[1:][iou <= threshold]

        return keep

    def _aggregate_per_class(self, detections_per_batch):
        """
        Aggregate detections to get top-1 per class format.
        Returns: confidences [K, B], boxes [K, B, 4]
        """
        B = len(detections_per_batch)
        K = self.num_classes

        confidences = np.zeros((K, B), dtype=np.float32)
        boxes = np.zeros((K, B, 4), dtype=np.float32)

        for b_idx, batch_dets in enumerate(detections_per_batch):
            if len(batch_dets) == 0:
                continue

            # Group by class and select highest confidence
            for det in batch_dets:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)

                if 0 <= cls_id < K and conf > confidences[cls_id, b_idx]:
                    confidences[cls_id, b_idx] = conf
                    boxes[cls_id, b_idx] = [x1, y1, x2, y2]

        return confidences, boxes

    def __call__(self, images, conf_threshold=0.25) -> Tuple[np.ndarray, np.ndarray]:
        """Return confidences [K, B] and boxes [K, B, 4]."""
        # Preprocess images to ONNX input format
        inp, metas = preprocess_images(images)

        # Run ONNX inference
        input_name = self.sess.get_inputs()[0].name
        raw_output = self.sess.run(None, {input_name: inp})[0]  # [B, 6, 8400]

        # Postprocess ONNX output to get filtered detections
        detections_per_batch = self._postprocess_onnx_output(raw_output, conf_threshold)

        # Convert boxes back to original image space
        for b_idx, (batch_dets, meta) in enumerate(zip(detections_per_batch, metas)):
            if len(batch_dets) > 0:
                r, dw, dh = meta["r"], meta["dw"], meta["dh"]
                # Unletterbox the coordinates
                batch_dets[:, 0] = (batch_dets[:, 0] - dw) / r  # x1
                batch_dets[:, 1] = (batch_dets[:, 1] - dh) / r  # y1
                batch_dets[:, 2] = (batch_dets[:, 2] - dw) / r  # x2
                batch_dets[:, 3] = (batch_dets[:, 3] - dh) / r  # y2

        # Aggregate to per-class format
        confidences, boxes = self._aggregate_per_class(detections_per_batch)

        return confidences, boxes


def plot_detections(images, confidences, boxes, class_names=None, score_thresh=0.1):
    """Plot detections on images."""
    B = confidences.shape[1]
    K = confidences.shape[0]
    colors = plt.cm.get_cmap('Set3', K)

    for b in range(B):  # Plot first 3 images max
        img = images[b]
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        detected_objects = 0
        for k in range(K):
            conf = confidences[k, b]
            if conf < score_thresh:
                continue

            detected_objects += 1
            x1, y1, x2, y2 = boxes[k, b]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=colors(k), facecolor='none')
            ax.add_patch(rect)

            class_name = class_names[k] if class_names else f"Class {k}"
            label = f"{class_name}: {conf:.3f}"
            ax.text(x1, max(0, y1 - 5), label, color=colors(k), fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        ax.set_title(f"Image {b} - {detected_objects} detections")
        ax.axis('off')
        plt.tight_layout()
        plt.show()


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


if __name__ == "__main__":
    # Configuration
    checkpoint_path = "checkpoints/yolo_fetal_structures_20250923_223741/weights/best.pt"
    dataset_dir = os.getenv("DATASET_DIR", "data")
    img_path_template = f"{dataset_dir}/Images/pd_{{}}.png"

    # Load model and create inference engines
    model = YOLO(checkpoint_path, task='detect')

    # Save to ONNX, then load it with onnxruntime, and compare results
    yolo_onnx_path = model.export(format="onnx", dynamic=True, opset=17)  # Export
    assert os.path.isfile(yolo_onnx_path), f"ONNX model not found at {yolo_onnx_path}"
    onnx_inference = TwoStageONNXInference(yolo_onnx_path, len(model.names))  # Pass num_classes instead of model
    baseline_inference = BaselineEncapsulatedYOLO(model)
    class_names = model.names
    print(f"Model has {len(class_names)} classes: {class_names}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using ONNX model: {yolo_onnx_path}")
    print(f"Using dataset directory: {dataset_dir}")
    print()

    # Load sample images
    ids = [109, 477, 398, 468, 469]
    images = []
    for img_id in ids:
        img_path = img_path_template.format(img_id)
        assert os.path.isfile(img_path), f"Image not found: {img_path}"
        img = cv2.imread(img_path)
        images.append(img)
    print(f"Loaded {len(images)} images for inference.")
    print()

    # Run baseline and ONNX inference
    baseline_conf, baseline_boxes = baseline_inference(images)
    onnx_conf, onnx_boxes = onnx_inference(images)
    print("Inference completed.\n")

    # Compare outputs
    comparison_passed = compare_outputs(baseline_conf, baseline_boxes, onnx_conf, onnx_boxes, class_names)
    print()

    # Visualize results
    print("Side-by-Side Comparison:")
    plot_side_by_side_comparison(images, baseline_conf, baseline_boxes, onnx_conf, onnx_boxes, class_names,
                                 score_thresh=0.1)
