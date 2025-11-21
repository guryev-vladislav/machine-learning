# tests/test_data_loaders.py
import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.data_loaders.deepcrack_loader import DeepCrackDataset
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    @staticmethod
    def create_test_image(path, size=(100, 100)):
        img = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def test_deepcrack_loader(self):
        data_dir = Path(self.temp_dir)
        (data_dir / "train" / "images").mkdir(parents=True)
        (data_dir / "train" / "masks").mkdir(parents=True)

        self.create_test_image(data_dir / "train" / "images" / "test1.jpg")
        self.create_test_image(data_dir / "train" / "masks" / "test1.jpg")

        dataset = DeepCrackDataset(data_dir, is_train=True)
        self.assertEqual(len(dataset), 1)

        image, mask = dataset[0]
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(mask.shape, (512, 512))


if __name__ == '__main__':
    unittest.main()