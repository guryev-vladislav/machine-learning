# src/data_loaders/base_loader.py
import os
import logging
from torch.utils.data import Dataset
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseCrackDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.masks = []

    def _discover_files(self):
        raise NotImplementedError("Subclasses must implement _discover_files")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            mask_path = self.masks[idx]

            image = self._load_image(image_path)
            mask = self._load_mask(mask_path)

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            return image, mask

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise

    def _load_image(self, image_path):
        raise NotImplementedError("Subclasses must implement _load_image")

    def _load_mask(self, mask_path):
        raise NotImplementedError("Subclasses must implement _load_mask")