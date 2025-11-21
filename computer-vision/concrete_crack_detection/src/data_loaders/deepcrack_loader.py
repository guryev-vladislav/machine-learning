# src/data_loaders/deepcrack_loader.py
import cv2
import numpy as np
from pathlib import Path
from .base_loader import BaseCrackDataset


class DeepCrackDataset(BaseCrackDataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        super().__init__(data_dir, transform, is_train)
        self._discover_files()

    def _discover_files(self):
        split_folder = "train" if self.is_train else "test"
        images_dir = self.data_dir / split_folder / "images"
        masks_dir = self.data_dir / split_folder / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise ValueError(f"DeepCrack dataset folders not found in {self.data_dir}")

        image_files = sorted(images_dir.glob("*.jpg"))
        mask_files = sorted(masks_dir.glob("*.jpg"))

        if len(image_files) != len(mask_files):
            raise ValueError("Mismatch between number of images and masks")

        self.images = image_files
        self.masks = mask_files

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (mask > 127).astype(np.float32)