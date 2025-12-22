import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
from .base_loader import BaseCrackDataset

logger = logging.getLogger(__name__)


class UnifiedCrackDataset(BaseCrackDataset):
    def __init__(self, crack_dir, non_crack_dir, is_train=True, target_size=(256, 256), task='segmentation'):
        super().__init__(crack_dir, is_train=is_train)
        self.target_size = target_size
        self.non_crack_dir = Path(non_crack_dir)
        self.crack_dir = Path(crack_dir)
        self.task = task  # 'segmentation' или 'classification'

        # Настройка аугментаций (важно для сегментации использовать одинаковые для img и mask)
        if self.is_train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=target_size, scale=(0.8, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        self._discover_unified_files()

    def _discover_unified_files(self):
        self.images, self.masks, self.labels = [], [], []

        # 1. Загружаем DeepCrack (основной источник для сегментации)
        img_dir = self.crack_dir / "rgb"
        mask_dir = self.crack_dir / "BW"

        crack_imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        for p in crack_imgs:
            # Ищем маску с таким же именем
            m = mask_dir / f"{p.stem}.png"
            if not m.exists():
                m = mask_dir / f"{p.stem}.jpg"

            if m.exists():
                self.images.append(p)
                self.masks.append(m)
                self.labels.append(1.0)

        crack_count = len(self.images)

        # 2. Добавляем пустые изображения ТОЛЬКО для классификации
        # Если мы учим UNet, нам нужны только картинки с реальными масками трещин
        if self.task == 'classification':
            non_crack_imgs = list(self.non_crack_dir.rglob("*Non-cracked/*.jpg"))
            if not non_crack_imgs:
                non_crack_imgs = list(self.non_crack_dir.rglob("*.jpg"))

            if crack_count > 0 and len(non_crack_imgs) > 0:
                np.random.seed(42)
                # Для классификации делаем баланс 1:1
                num_to_select = min(crack_count, len(non_crack_imgs))
                selected_indices = np.random.choice(len(non_crack_imgs), num_to_select, replace=False)

                for idx in selected_indices:
                    self.images.append(non_crack_imgs[idx])
                    self.masks.append(None)
                    self.labels.append(0.0)

                logger.info(f"Classification Mode: {crack_count} cracks + {num_to_select} background.")
        else:
            logger.info(f"Segmentation Mode: {crack_count} images with target masks loaded.")

        if len(self.images) == 0:
            raise RuntimeError(f"No images found! Check path: {self.crack_dir}")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Жесткая бинаризация: все что выше 127 -> 1.0, остальное 0.0
            mask = (mask > 127).astype(np.float32)
        else:
            # Если маски нет (для пустых фото в CNN), создаем пустую
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # Синхронная аугментация (Albumentations гарантирует, что кроп на картинке и маске совпадет)
        augmented = self.aug(image=image, mask=mask)

        # Возвращаем: Изображение, Маску (C,H,W) и Метку (0/1)
        return augmented['image'], augmented['mask'].unsqueeze(0), torch.tensor([self.labels[idx]], dtype=torch.float32)