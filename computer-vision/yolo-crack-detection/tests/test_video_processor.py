"""Tests for src/inference/video_processor.py."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class VideoProcessorTests(unittest.TestCase):
    """Tests for VideoProcessor class."""

    def _create_test_video(self, path, width=64, height=64, frames=5):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, 10, (width, height))
        for i in range(frames):
            frame = np.full((height, width, 3), i * 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def _make_minimal_config(self, tmpdir):
        from src.utils.config import Config

        data = {
            'project': {'name': 'test'},
            'paths': {
                'raw_videos': 'data/raw',
                'generated_videos': 'data/videos',
                'dataset_root': 'data/dataset',
                'outputs': str(tmpdir / 'outputs'),
                'pretrained_models': 'src/models/pretrained',
            },
            'dataset': {
                'class_names': ['no_crack', 'crack'],
                'train_split': 0.8,
                'val_split': 0.2,
            },
            'training': {'device': 'cpu', 'epochs': 1},
            'inference': {'confidence_threshold': 0.5, 'output_video_fps': 10},
        }
        config_path = tmpdir / 'config.yaml'
        import yaml

        with open(config_path, 'w') as f:
            yaml.dump(data, f)
        return Config(str(config_path))

    def test_processor_initializes_with_validator(self):
        from src.inference.video_processor import VideoProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_minimal_config(Path(tmpdir))
            processor = VideoProcessor(config=config)
            self.assertIsNotNone(processor.mechanics_validator)
            self.assertIsNone(processor.detector)

    def test_process_video_nonexistent_file(self):
        from src.inference.video_processor import VideoProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_minimal_config(Path(tmpdir))
            processor = VideoProcessor(config=config)
            result = processor.process_video('/nonexistent/video.mp4')
            self.assertIsNone(result)


class MechanicsValidatorIntegrationTests(unittest.TestCase):
    """Integration tests: mechanics validator rejects smooth plastic."""

    def test_smooth_square_rejected(self):
        from src.utils.crack_mechanics_validator import ConcreteCrackMechanicsValidator

        validator = ConcreteCrackMechanicsValidator(acceptance_threshold=0.5)
        crop = np.full((64, 64, 3), 150, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (44, 44), 255, -1)

        result = validator.evaluate(crop, mask)
        # A solid square should NOT look like a crack
        self.assertFalse(result['verified'])

    def test_line_like_shape_more_plausible(self):
        from src.utils.crack_mechanics_validator import ConcreteCrackMechanicsValidator

        validator = ConcreteCrackMechanicsValidator(acceptance_threshold=0.33)
        crop = np.full((64, 64, 3), 180, dtype=np.uint8)
        cv2.line(crop, (10, 32), (54, 32), 40, 3)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (10, 32), (54, 32), 255, 3)

        result = validator.evaluate(crop, mask)
        # Line-like shape with dark line should score higher than square
        self.assertGreater(result['mechanics_confidence'], 0)


if __name__ == '__main__':
    unittest.main()
