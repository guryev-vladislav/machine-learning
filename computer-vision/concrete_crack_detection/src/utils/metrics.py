# src/utils/metrics.py
import logging

import torch

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, predictions, targets):
        try:
            predictions = (predictions > self.threshold).float()
            targets = (targets > self.threshold).float()

            self.tp += torch.sum((predictions == 1) & (targets == 1)).item()
            self.fp += torch.sum((predictions == 1) & (targets == 0)).item()
            self.fn += torch.sum((predictions == 0) & (targets == 1)).item()
            self.tn += torch.sum((predictions == 0) & (targets == 0)).item()

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def compute_iou(self):
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        return intersection / union if union > 0 else 0.0

    def compute_dice(self):
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn) if (2 * self.tp + self.fp + self.fn) > 0 else 0.0

    def compute_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def compute_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def compute_f1(self):
        precision = self.compute_precision()
        recall = self.compute_recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def get_all_metrics(self):
        return {
            'iou': self.compute_iou(),
            'dice': self.compute_dice(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1': self.compute_f1(),
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn
        }