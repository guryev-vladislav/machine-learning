import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    import torch
    from pathlib import Path
    from .base_loader import BaseCrackDataset
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

class SDNETDataset(BaseCrackDataset):
    def __init__(self, data_dir, transform=None, is_train=True, target_size=(256, 256)):
        self.target_size = target_size
        super().__init__(data_dir, transform, is_train)
        self._discover_files()

    def _discover_files(self):
        object_types = ["Decks", "Pavements", "Walls"]
        crack_images, no_crack_images = [], []

        for obj_type in object_types:
            base_path = self.data_dir / obj_type
            c_dir, n_dir = base_path / "Cracked", base_path / "Non-cracked"
            if c_dir.exists():
                crack_images.extend(list(c_dir.glob("*.jpg")))
            if n_dir.exists():
                no_crack_images.extend(list(n_dir.glob("*.jpg")))

        self.images = crack_images + no_crack_images
        self.labels = [1.0] * len(crack_images) + [0.0] * len(no_crack_images)
        self.masks = self.images

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Image not loaded: {image_path}")
            raise FileNotFoundError(f"Image not loaded: {image_path}")
        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        try:
            idx = self.images.index(mask_path)
            return np.array([self.labels[idx]], dtype=np.float32)
        except ValueError as e:
            logger.error(f"Mask/Label not found for path: {mask_path}")
            raise

    def __getitem__(self, idx):
        img = self._load_image(self.images[idx])
        label = self._load_mask(self.masks[idx])
        return torch.from_numpy(img), torch.from_numpy(label)