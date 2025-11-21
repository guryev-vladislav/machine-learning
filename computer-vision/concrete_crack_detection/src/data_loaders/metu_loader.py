# src/data_loaders/metu_loader.py
import cv2
import numpy as np

from .base_loader import BaseCrackDataset


class METUDataset(BaseCrackDataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        super().__init__(data_dir, transform, is_train)
        self._discover_files()

    def _discover_files(self):
        images_dir = self.data_dir / "Images"
        masks_dir = self.data_dir / "Masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise ValueError(f"METU dataset folders not found in {self.data_dir}")

        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        mask_files = sorted(masks_dir.glob("*.jpg")) + sorted(masks_dir.glob("*.png"))

        if len(image_files) != len(mask_files):
            raise ValueError("Mismatch between number of images and masks")

        split_idx = int(0.8 * len(image_files))

        if self.is_train:
            self.images = image_files[:split_idx]
            self.masks = mask_files[:split_idx]
        else:
            self.images = image_files[split_idx:]
            self.masks = mask_files[split_idx:]

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.float32)