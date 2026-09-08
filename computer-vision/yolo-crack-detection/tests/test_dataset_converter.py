import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.data_preparation.dataset_converter import DatasetConverter


class DatasetConverterTests(unittest.TestCase):
    def test_split_is_reproducible_and_shuffled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SimpleNamespace(
                DATASET_ROOT=root / 'dataset',
                RANDOM_SEED=42,
                TRAIN_SPLIT=0.75,
            )
            source = root / 'source'
            source.mkdir()
            image_paths = []
            for index in range(4):
                image_path = source / f'image_{index}.jpg'
                image_path.write_bytes(b'image')
                image_paths.append(image_path)

            converter = DatasetConverter(config)
            converter._split_and_copy(image_paths, 'crack')

            train_files = sorted((config.DATASET_ROOT / 'train' / 'crack').iterdir())
            val_files = sorted((config.DATASET_ROOT / 'val' / 'crack').iterdir())

            self.assertEqual(len(train_files), 3)
            self.assertEqual(len(val_files), 1)
            self.assertNotEqual([path.name for path in train_files], [path.name for path in image_paths[:3]])


if __name__ == '__main__':
    unittest.main()
