import os
import logging
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseCrackDataset(Dataset):
    """
    Базовый класс для всех датасетов по трещинам.
    Обеспечивает базовую логику инициализации, длины и получения элемента.
    """

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.masks = []  # Используется как маска для сегментации ИЛИ метка класса для классификации
        self.labels = []  # Дополнительное поле для классификации (используется в SDNETDataset)

        # Вызов _discover_files должен быть в дочернем классе после инициализации всех параметров
        # self._discover_files()

    def _discover_files(self):
        """
        Должен быть реализован в дочернем классе для заполнения self.images и self.masks/self.labels
        """
        raise NotImplementedError("Subclasses must implement _discover_files")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]

            # Для сегментации mask_path — это путь к маске, для классификации — индекс/путь к метке
            mask_path = self.masks[idx]

            image = self._load_image(image_path)
            # Загрузка может вернуть либо маску (H, W), либо метку (0 или 1)
            mask_or_label = self._load_mask(mask_path)

            if self.transform:
                # В случае классификации Albumentations не используется
                if isinstance(mask_or_label, np.ndarray) and mask_or_label.ndim > 0:
                    # Сегментация: трансформация применяется к изображению и маске
                    transformed = self.transform(image=image, mask=mask_or_label)
                    image = transformed['image']
                    mask_or_label = transformed['mask']
                else:
                    # Классификация: трансформация только для изображения
                    transformed = self.transform(image=image)
                    image = transformed['image']

            return image, mask_or_label

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # В реальном проекте здесь можно возвращать None и фильтровать None в DataLoader
            raise

    def _load_image(self, image_path):
        raise NotImplementedError("Subclasses must implement _load_image")

    def _load_mask(self, mask_path):
        raise NotImplementedError("Subclasses must implement _load_mask")