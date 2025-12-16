import logging
import cv2
import numpy as np
from pathlib import Path

from .base_loader import BaseCrackDataset

logger = logging.getLogger(__name__)


class SDNETDataset(BaseCrackDataset):
    """
    Загрузчик для классификационного датасета SDNET2018.
    Использует структуру {Decks, Pavements, Walls}/{Cracked, Non-cracked}.
    """

    def __init__(self, data_dir, transform=None, is_train=True):
        super().__init__(data_dir, transform, is_train)
        self._discover_files()

    def _discover_files(self):
        object_types = ["Decks", "Pavements", "Walls"]

        crack_images = []
        no_crack_images = []

        for obj_type in object_types:
            base_path = self.data_dir / obj_type

            crack_dir = base_path / "Cracked"
            if crack_dir.exists():
                crack_images.extend(list(crack_dir.glob("*.jpg")) + list(crack_dir.glob("*.png")))

            no_crack_dir = base_path / "Non-cracked"
            if no_crack_dir.exists():
                no_crack_images.extend(list(no_crack_dir.glob("*.jpg")) + list(no_crack_dir.glob("*.png")))

        self.images = crack_images + no_crack_images
        self.labels = [1] * len(crack_images) + [0] * len(no_crack_images)

        # ИСПРАВЛЕНИЕ: self.masks должен содержать пути, а не метки.
        # Используем список путей к изображениям как заглушку.
        self.masks = self.images

        if not self.images:
            raise ValueError(f"SDNET dataset folders not found or empty in {self.data_dir}")

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None: raise FileNotFoundError(f"Image not loaded: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        """
        Для классификации возвращает метку класса (0 или 1), а не маску.
        """
        # Находим индекс текущего пути (mask_path) в self.images, чтобы получить метку
        try:
            idx = self.images.index(mask_path)
            # Возвращаем метку, а не маску
            return self.labels[idx]
        except ValueError:
            raise ValueError(f"Internal Error: Path {mask_path} not found in images list.")