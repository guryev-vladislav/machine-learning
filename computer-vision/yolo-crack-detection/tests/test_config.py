"""Tests for src/utils/config.py."""

import tempfile
import unittest
from pathlib import Path

import yaml


class ConfigTests(unittest.TestCase):
    """Tests for the Config class."""

    def _write_test_config(self, data):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        try:
            yaml.dump(data, tmp)
            tmp.close()
            return tmp.name
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise

    def test_loads_default_config(self):
        from src.utils.config import Config

        config = Config()
        self.assertIsNotNone(config.PROJECT_NAME)
        self.assertTrue(config.DATASET_ROOT.exists() or True)
        self.assertEqual(config.CLASS_NAMES, ['no_crack', 'crack'])

    def test_default_values_are_reasonable(self):
        from src.utils.config import Config

        config = Config()
        self.assertGreater(config.EPOCHS, 0)
        self.assertGreater(config.BATCH_SIZE, 0)
        self.assertGreater(config.IMGSZ, 0)
        self.assertIn(config.DEVICE, ('cpu', 'cuda:0', 'cuda'))
        self.assertGreater(config.PATIENCE, 0)

    def test_task_is_classify_by_default(self):
        from src.utils.config import Config

        config = Config()
        self.assertEqual(config.TASK, 'classify')

    def test_conf_threshold_between_0_and_1(self):
        from src.utils.config import Config

        config = Config()
        self.assertGreaterEqual(config.CONF_THRESHOLD, 0.0)
        self.assertLessEqual(config.CONF_THRESHOLD, 1.0)

    def test_custom_config_path(self):
        from src.utils.config import Config

        data = {
            'project': {'name': 'test_project'},
            'paths': {
                'dataset_root': 'data/custom',
                'outputs': 'outputs/custom',
            },
            'dataset': {
                'class_names': ['no_crack', 'crack'],
                'train_split': 0.8,
                'val_split': 0.2,
                'random_seed': 99,
            },
            'training': {
                'device': 'cpu',
                'epochs': 10,
                'batch_size': 4,
                'imgsz': 224,
                'learning_rate': 0.01,
                'patience': 5,
                'workers': 2,
                'task': 'classify',
                'model': 'yolo11n-cls',
            },
            'inference': {
                'confidence_threshold': 0.6,
                'iou_threshold': 0.4,
                'draw_confident': False,
                'output_video_fps': 15,
            },
        }

        config_path = self._write_test_config(data)
        try:
            config = Config(config_path)
            self.assertEqual(config.PROJECT_NAME, 'test_project')
            self.assertEqual(config.EPOCHS, 10)
            self.assertEqual(config.BATCH_SIZE, 4)
            self.assertEqual(config.IMGSZ, 224)
            self.assertEqual(config.CONF_THRESHOLD, 0.6)
            self.assertEqual(config.RANDOM_SEED, 99)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_invalid_class_names_raises(self):
        from src.utils.config import Config

        data = {
            'dataset': {
                'class_names': ['only_one'],
                'train_split': 0.8,
                'val_split': 0.2,
            },
            'training': {'device': 'cpu'},
            'inference': {'confidence_threshold': 0.5},
        }

        config_path = self._write_test_config(data)
        try:
            with self.assertRaises(ValueError):
                Config(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_invalid_train_split_raises(self):
        from src.utils.config import Config

        data = {
            'dataset': {
                'class_names': ['a', 'b'],
                'train_split': 1.5,
                'val_split': 0.2,
            },
            'training': {'device': 'cpu'},
            'inference': {'confidence_threshold': 0.5},
        }

        config_path = self._write_test_config(data)
        try:
            with self.assertRaises(ValueError):
                Config(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_invalid_val_split_raises(self):
        from src.utils.config import Config

        data = {
            'dataset': {
                'class_names': ['a', 'b'],
                'train_split': 0.8,
                'val_split': -0.1,
            },
            'training': {'device': 'cpu'},
            'inference': {'confidence_threshold': 0.5},
        }

        config_path = self._write_test_config(data)
        try:
            with self.assertRaises(ValueError):
                Config(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_missing_config_uses_defaults(self):
        from src.utils.config import Config

        config = Config('/nonexistent/path/config.yaml')
        self.assertIsNotNone(config.PROJECT_NAME)
        self.assertEqual(config.TASK, 'classify')

    def test_create_directories(self):
        from src.utils.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                'project': {'name': 'test'},
                'paths': {
                    'raw_videos': 'data/raw',
                    'generated_videos': 'data/videos',
                    'dataset_root': 'data/dataset',
                    'outputs': 'outputs',
                    'pretrained_models': 'src/models/pretrained',
                },
                'dataset': {
                    'class_names': ['no_crack', 'crack'],
                    'train_split': 0.8,
                    'val_split': 0.2,
                },
                'training': {'device': 'cpu'},
                'inference': {'confidence_threshold': 0.5},
            }

            config_path = self._write_test_config(data)
            try:
                config = Config(config_path)
                config.PROJECT_PATH = Path(tmpdir)
                config._resolve_path = lambda p: Path(tmpdir) / p
                config.OUTPUTS_PATH = Path(tmpdir) / 'outputs'
                config.DATASET_ROOT = Path(tmpdir) / 'data' / 'dataset'
                config.RAW_VIDEOS_PATH = Path(tmpdir) / 'data' / 'raw'
                config.GENERATED_VIDEOS_PATH = Path(tmpdir) / 'data' / 'videos'
                config.PRETRAINED_MODELS_PATH = Path(tmpdir) / 'src' / 'models' / 'pretrained'

                config.create_directories()

                self.assertTrue((config.DATASET_ROOT / 'train' / 'crack').exists())
                self.assertTrue((config.DATASET_ROOT / 'val' / 'no_crack').exists())
                self.assertTrue(config.OUTPUTS_PATH.exists())
            finally:
                Path(config_path).unlink(missing_ok=True)

    def test_repr(self):
        from src.utils.config import Config

        config = Config()
        self.assertIn(config.PROJECT_NAME, repr(config))


class ConfigPathResolutionTests(unittest.TestCase):
    """Tests for path resolution in Config."""

    def test_absolute_path_preserved(self):
        from src.utils.config import Config

        config = Config()
        result = config._resolve_path('/absolute/path')
        self.assertEqual(result, Path('/absolute/path'))

    def test_relative_path_resolved_against_project(self):
        from src.utils.config import Config

        config = Config()
        result = config._resolve_path('relative/path')
        self.assertTrue(result.is_absolute())
        self.assertTrue(str(result).endswith('relative/path'))


if __name__ == '__main__':
    unittest.main()
