import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    import random
    import torch
    from pathlib import Path
    from .base_loader import BaseCrackDataset
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

class DeepCrackDataset(BaseCrackDataset):
    def __init__(self, data_dir, transform=None, is_train=True, train_ratio=0.8, seed=42, target_size=(256, 256)):
        self.train_ratio = train_ratio
        self.seed = seed
        self.target_size = target_size
        super().__init__(data_dir, transform, is_train)
        self._discover_files()

    def _discover_files(self):
        images_root = self.data_dir / "rgb"
        masks_root = self.data_dir / "BW"

        if not images_root.exists() or not masks_root.exists():
            err_msg = f"DeepCrack folders not found in {self.data_dir}"
            logger.critical(err_msg)
            raise ValueError(err_msg)

        all_image_paths = sorted(list(images_root.glob("*.jpg")) + list(images_root.glob("*.png")))
        stems = [p.stem for p in all_image_paths]

        random.seed(self.seed)
        random.shuffle(stems)
        split_idx = int(len(stems) * self.train_ratio)
        current_stems = stems[:split_idx] if self.is_train else stems[split_idx:]

        self.images = [images_root / f"{stem}{p.suffix}" for stem in current_stems for p in all_image_paths if p.stem == stem]
        self.masks = [masks_root / f"{stem}{p.suffix}" for stem in current_stems for p in all_image_paths if p.stem == stem]

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
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Mask not loaded: {mask_path}")
            raise FileNotFoundError(f"Mask not loaded: {mask_path}")
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)

    def __getitem__(self, idx):
        img = self._load_image(self.images[idx])
        mask = self._load_mask(self.masks[idx])
        return torch.from_numpy(img), torch.from_numpy(mask)