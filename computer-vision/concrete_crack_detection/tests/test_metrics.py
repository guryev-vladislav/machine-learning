# tests/test_metrics.py
import unittest
import torch
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.metrics import SegmentationMetrics
except ImportError as e:
    print(f"Import error: {e}")
    raise

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = SegmentationMetrics(threshold=0.5)

    def test_perfect_prediction(self):
        predictions = torch.ones(2, 1, 10, 10)
        targets = torch.ones(2, 1, 10, 10)
        self.metrics.update(predictions, targets)
        results = self.metrics.get_all_metrics()
        self.assertEqual(results['iou'], 1.0)
        self.assertEqual(results['dice'], 1.0)

    def test_no_overlap(self):
        predictions = torch.ones(2, 1, 10, 10)
        targets = torch.zeros(2, 1, 10, 10)
        self.metrics.update(predictions, targets)
        results = self.metrics.get_all_metrics()
        self.assertEqual(results['iou'], 0.0)
        self.assertEqual(results['dice'], 0.0)

    def test_reset(self):
        predictions = torch.ones(2, 1, 10, 10)
        targets = torch.ones(2, 1, 10, 10)
        self.metrics.update(predictions, targets)
        self.metrics.reset()
        results = self.metrics.get_all_metrics()
        self.assertEqual(results['tp'], 0)

if __name__ == '__main__':
    unittest.main()