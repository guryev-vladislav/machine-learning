import yaml
import torch
from pathlib import Path


class Config:
    def __init__(self, config_path="configs/train_config.yaml"):
        with open(config_path, 'r') as f:
            self.data = yaml.safe_load(f)

        # Основные пути
        self.BASE_PATH = Path(self.data['project']['base_path'])
        self.DEEPCRACK_PATH = self.BASE_PATH / self.data['paths']['deepcrack']
        self.SDNET_PATH = self.BASE_PATH / self.data['paths']['sdnet']
        self.CHECKPOINT_DIR = self.BASE_PATH / self.data['paths']['checkpoints']

        # Пути к весам
        self.UNET_WEIGHTS = self.CHECKPOINT_DIR / self.data['models']['unet']['weights_name']
        self.CNN_WEIGHTS = self.CHECKPOINT_DIR / self.data['models']['cnn']['weights_name']

        # Параметры обучения
        self.DEVICE = torch.device(self.data['training']['device'] if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = self.data['training']['batch_size']
        self.LR = float(self.data['training']['learning_rate'])
        self.EPOCHS = self.data['training']['epochs']
        self.NUM_WORKERS = self.data['training']['num_workers']

        # Размеры изображений
        self.PATCH_SIZE = self.data['models']['unet']['patch_size']
        self.IMAGE_SIZE = self.data['models']['cnn']['image_size']

        # Создаем папку для чекпоинтов, если её нет
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Config(Loaded from YAML, Device: {self.DEVICE}, Epochs: {self.EPOCHS})"