import tempfile
import unittest
from pathlib import Path

from scripts.create_smoke_dataset import create_dataset


class SmokeDatasetTests(unittest.TestCase):
    def test_create_dataset_writes_expected_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'dataset'
            create_dataset(output_path, train_count=2, val_count=1, size=32, seed=42)

            for split, count in (('train', 2), ('val', 1)):
                for class_name in ('crack', 'no_crack'):
                    images = list((output_path / split / class_name).glob('*.png'))
                    self.assertEqual(len(images), count)


if __name__ == '__main__':
    unittest.main()
