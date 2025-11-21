# src/data_loaders/sdnet_loader.py
import logging

import cv2
import numpy as np

from .base_loader import BaseCrackDataset

logger = logging.getLogger(__name__)


class SDNETDataset(BaseCrackDataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        super().__init__(data_dir, transform, is_train)
        self._discover_files()

    def _discover_files(self):
        crack_dir = self.data_dir / "C"
        no_crack_dir = self.data_dir / "U"

        crack_images = list(crack_dir.glob("*.jpg")) + list(crack_dir.glob("*.png"))
        no_crack_images = list(no_crack_dir.glob("*.jpg")) + list(no_crack_dir.glob("*.png"))

        self.images = crack_images + no_crack_images
        self.labels = [1] * len(crack_images) + [0] * len(no_crack_images)

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        return self.labels[self.images.index(mask_path)]

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            label = self.labels[idx]

            image = self._load_image(image_path)

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            return image, label

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise