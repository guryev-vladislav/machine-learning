import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml
    import torch
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

class Config:
    def __init__(self, config_path="configs/train_config.yaml"):
        with open(config_path, 'r') as f:
            self.data = yaml.safe_load(f)

        self.BASE_PATH = Path(self.data['project']['base_path'])
        self.DEEPCRACK_PATH = self.BASE_PATH / self.data['paths']['deepcrack']
        self.SDNET_PATH = self.BASE_PATH / self.data['paths']['sdnet']
        self.CHECKPOINT_DIR = self.BASE_PATH / self.data['paths']['checkpoints']

        self.DEVICE = torch.device(self.data['training']['device'] if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = self.data['training']['batch_size']
        self.LR = float(self.data['training']['learning_rate'])
        self.EPOCHS = self.data['training']['epochs']
        self.NUM_WORKERS = self.data['training']['num_workers']
        self.POS_WEIGHT_CNN = self.data['training']['pos_weight_cnn']
        self.POS_WEIGHT_UNET = self.data['training']['pos_weight_unet']
        self.EARLY_STOPPING = self.data['training']['early_stopping']

        self.PATCH_SIZE = self.data['models']['unet']['patch_size']
        self.IMAGE_SIZE = self.data['models']['cnn']['image_size']
        self.WEIGHT_NAME_UNET = self.data['models']['unet']['weights_name']
        self.WEIGHT_NAME_CNN = self.data['models']['cnn']['weights_name']

    def __repr__(self):
        return f"Config(Loaded from YAML, LR: {self.LR}, Epochs: {self.EPOCHS})"