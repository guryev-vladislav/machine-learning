import cv2
import numpy as np
import random
from pathlib import Path
from .base_loader import BaseCrackDataset


class DeepCrackDataset(BaseCrackDataset):
    def __init__(self, data_dir, transform=None, is_train=True, train_ratio=0.8, seed=42):
        self.train_ratio = train_ratio
        self.seed = seed
        super().__init__(data_dir, transform, is_train)
        self._discover_files()  # Теперь вызывается здесь

    def _discover_files(self):
        images_root = self.data_dir / "rgb"
        masks_root = self.data_dir / "BW"

        if not images_root.exists() or not masks_root.exists():
            raise ValueError(f"DeepCrack dataset folders not found in {self.data_dir}. Expected 'rgb' and 'BW'.")

        # 1. Собираем все пути к изображениям и извлекаем stem-имена
        all_image_paths = sorted(list(images_root.glob("*.jpg")) + list(images_root.glob("*.png")))
        stems = [p.stem for p in all_image_paths]

        if not stems:
            raise ValueError("No images found in DeepCrack dataset.")

        # 2. Выполняем детерминированный сплит
        random.seed(self.seed)
        random.shuffle(stems)

        split_idx = int(len(stems) * self.train_ratio)

        if self.is_train:
            current_stems = stems[:split_idx]
        else:
            current_stems = stems[split_idx:]

        # 3. Формируем списки изображений и масок
        self.images = [images_root / f"{stem}{p.suffix}" for stem in current_stems for p in all_image_paths if
                       p.stem == stem]
        self.masks = [masks_root / f"{stem}{p.suffix}" for stem in current_stems for p in all_image_paths if
                      p.stem == stem]

        if len(self.images) != len(self.masks):
            raise ValueError("Mismatch between number of images and masks after split")

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None: raise FileNotFoundError(f"Image not loaded: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(f"Mask not loaded: {mask_path}")
        return (mask > 127).astype(np.float32)