import os
import sys
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path=None):
        # PROJECT_PATH это папка yolo-crack-detection
        # src/utils/config.py -> utils -> src -> yolo-crack-detection
        self.PROJECT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.resolve()

        # Загружаем config из YAML если передан
        if config_path is None:
            config_path = self.PROJECT_PATH / "configs" / "train_config.yaml"

        if isinstance(config_path, str):
            config_path = Path(config_path)

        self.config_path = config_path
        self.config_dict = {}

        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config_dict = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")

        self._load_from_dict()

    def _load_from_dict(self):
        """Загружаем параметры из словаря конфига"""
        cfg = self.config_dict

        # Project
        project = cfg.get('project', {})
        self.PROJECT_NAME = project.get('name', 'yolo_crack_detection')

        # Paths
        paths = cfg.get('paths', {})
        self.RAW_VIDEOS_PATH = self._resolve_path(paths.get('raw_videos', 'data/raw'))
        self.GENERATED_VIDEOS_PATH = self._resolve_path(paths.get('generated_videos', 'data/videos'))
        self.DATASET_ROOT = self._resolve_path(paths.get('dataset_root', 'data/dataset'))
        self.DEEPCRACK_PATH = self._resolve_path(paths.get('deepcrack', 'data/external/deepcrack'))
        self.SDNET_PATH = self._resolve_path(paths.get('sdnet', 'data/external/sdnet2018'))
        self.OUTPUTS_PATH = self._resolve_path(paths.get('outputs', 'outputs'))

        # Video settings
        video = cfg.get('video', {})
        self.VIDEO_FPS = video.get('fps', 10)
        self.VIDEO_WIDTH = video.get('width', 640)
        self.VIDEO_HEIGHT = video.get('height', 480)
        self.VIDEO_CODEC = video.get('codec', 'mp4v')

        # Dataset
        dataset = cfg.get('dataset', {})
        self.TRAIN_SPLIT = dataset.get('train_split', 0.8)
        self.VAL_SPLIT = dataset.get('val_split', 0.2)
        self.RANDOM_SEED = dataset.get('random_seed', 42)
        self.CLASS_NAMES = dataset.get('class_names', ['no_crack', 'crack'])

        # Training
        training = cfg.get('training', {})
        self.DEVICE = training.get('device', 'cuda:0')
        self.EPOCHS = training.get('epochs', 50)
        self.BATCH_SIZE = training.get('batch_size', 32)
        self.IMGSZ = training.get('imgsz', 640)
        self.LR = training.get('learning_rate', 0.001)
        self.PATIENCE = training.get('patience', 10)
        self.NUM_WORKERS = training.get('workers', 4)
        self.TASK = training.get('task', 'classify')
        self.MODEL_NAME = training.get('model', 'yolov8n-cls')

        # Inference
        inference = cfg.get('inference', {})
        self.CONF_THRESHOLD = inference.get('confidence_threshold', 0.5)
        self.IOU_THRESHOLD = inference.get('iou_threshold', 0.5)
        self.DRAW_CONFIDENT = inference.get('draw_confident', True)
        self.OUTPUT_VIDEO_FPS = inference.get('output_video_fps', 20)

    def _resolve_path(self, path_str):
        """Резолвим пути относительно PROJECT_PATH"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.PROJECT_PATH / path

    def create_directories(self):
        """Создаём все необходимые директории"""
        dirs = [
            self.RAW_VIDEOS_PATH,
            self.GENERATED_VIDEOS_PATH,
            self.DATASET_ROOT,
            self.OUTPUTS_PATH,
            self.DATASET_ROOT / 'train' / 'crack',
            self.DATASET_ROOT / 'train' / 'no_crack',
            self.DATASET_ROOT / 'val' / 'crack',
            self.DATASET_ROOT / 'val' / 'no_crack',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created/checked directory: {d}")

    def __repr__(self):
        return f"<Config: {self.PROJECT_NAME}>"

