# src/training/train_utils.py
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device

def save_model_architecture(model, model_path):
    try:
        arch_file = Path(model_path) / "model_architecture.txt"
        with open(arch_file, 'w') as f:
            f.write(str(model))
        logger.info(f"Model architecture saved to {arch_file}")
    except Exception as e:
        logger.error(f"Failed to save model architecture: {e}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)