# src/utils/transforms.py
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ToTensor:
    """Convert PIL Image or numpy array to tensor"""

    def __call__(self, sample):
        if isinstance(sample, (Image.Image, np.ndarray)):
            # Convert to tensor and normalize to [0,1]
            if isinstance(sample, Image.Image):
                sample = np.array(sample)

            # Handle different input types
            if sample.ndim == 2:  # Grayscale
                sample = sample[np.newaxis, :, :]
            elif sample.ndim == 3:  # RGB
                sample = sample.transpose(2, 0, 1)

            sample = torch.from_numpy(sample).float() / 255.0

        return sample


class Normalize:
    """Normalize tensor with mean and std"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            return (sample - self.mean) / self.std
        return sample


class Resize:
    """Resize image to target size"""

    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, sample):
        if isinstance(sample, (Image.Image, np.ndarray)):
            if isinstance(sample, np.ndarray):
                sample = Image.fromarray(sample)
            sample = sample.resize(self.size, Image.BILINEAR)
            sample = np.array(sample)
        return sample


class Compose:
    """Compose several transforms together"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transforms(image_size=(512, 512)):
    """Get training transforms without albumentations"""
    try:
        return Compose([
            Resize(image_size),
            ToTensor(),
            Normalize()
        ])
    except Exception as e:
        logger.error(f"Error creating train transforms: {e}")
        raise


def get_val_transforms(image_size=(512, 512)):
    """Get validation transforms without albumentations"""
    try:
        return Compose([
            Resize(image_size),
            ToTensor(),
            Normalize()
        ])
    except Exception as e:
        logger.error(f"Error creating val transforms: {e}")
        raise


def get_test_transforms(image_size=(512, 512)):
    """Get test transforms without albumentations"""
    return get_val_transforms(image_size)