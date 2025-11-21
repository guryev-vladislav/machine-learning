# src/models/model_factory.py
import logging
import torch

from .unet import UNet
from .classifier import CrackClassifier

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        try:
            if model_type == "unet":
                return UNet(**kwargs)
            elif model_type == "classifier":
                return CrackClassifier(**kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            raise

    @staticmethod
    def load_model(model_path, model_type, **kwargs):
        try:
            model = ModelFactory.create_model(model_type, **kwargs)
            checkpoint = torch.load(model_path)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise