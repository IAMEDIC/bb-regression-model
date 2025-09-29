"""
YOLO ONNX Inference Module
Pure ONNX runtime inference with custom output format.
"""
import os
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib import patches


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


class YOLOONNXInference:
    """Pure ONNX inference: raw YOLO ONNX + pure NumPy postprocessing."""

    def __init__(self, onnx_path: str, num_classes: int):
        """
        Initialize ONNX inference engine.

        Args:
            onnx_path: Path to the ONNX model file
            num_classes: Number of classes in the model
        """
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
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
        Convert raw ONNX output [B, 4+num_classes, 8400] to filtered detections.
        For 2 classes: [B, 6, 8400] = [x, y, w, h, class0_conf, class1_conf]
        Returns: List of detection arrays, one per batch item.
        Each detection array has shape [N, 6] = [x1, y1, x2, y2, conf, class_id]
        """
        batch_size = raw_output.shape[0]
        detections_per_batch = []

        for b in range(batch_size):
            # Extract components from raw output [6, 8400]
            x_center = raw_output[b, 0, :]  # Channel 0: x center
            y_center = raw_output[b, 1, :]  # Channel 1: y center
            width = raw_output[b, 2, :]     # Channel 2: width
            height = raw_output[b, 3, :]    # Channel 3: height

            # Extract class confidences (channels 4 onwards are class confidences)
            class_confs = raw_output[b, 4:, :]  # [num_classes, 8400]

            # Find the class with maximum confidence for each anchor
            max_class_confs = np.max(class_confs, axis=0)  # [8400]
            predicted_classes = np.argmax(class_confs, axis=0)  # [8400]

            # Filter by confidence threshold
            valid_mask = max_class_confs > conf_threshold
            if not np.any(valid_mask):
                detections_per_batch.append(np.empty((0, 6)))
                continue

            # Apply mask to all components
            x_center = x_center[valid_mask]
            y_center = y_center[valid_mask]
            width = width[valid_mask]
            height = height[valid_mask]
            confidence = max_class_confs[valid_mask]
            class_id = predicted_classes[valid_mask]

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
        """
        Run inference on a batch of images.

        Args:
            images: List of images (numpy arrays in BGR format)
            conf_threshold: Confidence threshold for filtering detections

        Returns:
            Tuple of (confidences, boxes):
            - confidences: [K, B] array of confidence scores
            - boxes: [K, B, 4] array of bounding boxes in xyxy format
        """
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

    for b in range(B):
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


# Example usage
if __name__ == "__main__":
    # Example usage of the ONNX inference module
    onnx_path = "checkpoints/yolo_fetal_structures_20250923_223741/weights/best.onnx"
    num_classes = 2  # NT, NB
    class_names = {0: 'NT', 1: 'NB'}

    # Initialize inference engine
    inference_engine = YOLOONNXInference(onnx_path, num_classes)

    # Load sample images
    from dotenv import load_dotenv
    load_dotenv()
    dataset_dir = os.getenv("DATASET_DIR", "data")
    img_path_template = f"{dataset_dir}/Images/pd_{{}}.png"

    ids = [109, 477, 398, 468, 469]
    images = []
    for img_id in ids:
        img_path = img_path_template.format(img_id)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            images.append(img)

    if images:
        print(f"Running inference on {len(images)} images...")
        confidences, boxes = inference_engine(images)

        print(f"Output shapes: confidences {confidences.shape}, boxes {boxes.shape}")
        print("Confidences per class and image:")
        for k in range(num_classes):
            class_name = class_names.get(k, f"Class {k}")
            print(f"  {class_name}: {confidences[k]}")

        # Visualize results
        plot_detections(images, confidences, boxes, class_names, score_thresh=0.1)
    else:
        print("No images found. Please check your dataset path.")
