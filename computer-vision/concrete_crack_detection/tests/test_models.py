# tests/test_models.py
import unittest
import torch
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.unet import UNet
    from src.models.classifier import CrackClassifier
except ImportError as e:
    print(f"Import error: {e}")
    raise

class TestModels(unittest.TestCase):
    def test_unet_forward(self):
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        self.assertEqual(output.shape, (2, 1, 256, 256))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_classifier_forward(self):
        model = CrackClassifier(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))

    def test_unet_parameters(self):
        model = UNet()
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 1000000)

if __name__ == '__main__':
    unittest.main()