import os
import sys
import logging
from pathlib import Path
import random
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from src.utils.config import Config
except ImportError as e:
    logger.error(f"Critical import error: {e}")
    sys.exit(1)

class DatasetConverter:
    """Конвертирует датасет в YOLO Classification формат"""

    def __init__(self, config=None):
        self.config = config or Config()
        random.seed(self.config.RANDOM_SEED)

    def build_yolo_dataset(self):
        """Строит YOLO датасет из DeepCrack и SDNET"""
        logger.info("Building YOLO classification dataset...")

        self.config.create_directories()

        # Собираем все изображения с их лейблами
        crack_images = []
        non_crack_images = []

        # Из DeepCrack (считаем что все - трещины)
        logger.info("Collecting DeepCrack images (crack)...")
        rgb_dir = self.config.DEEPCRACK_PATH / "rgb"
        if rgb_dir.exists():
            images = self._get_images_from_dir(rgb_dir)
            crack_images.extend(images)
            logger.info(f"  Found {len(images)} crack images")

        # Из SDNET cracked
        logger.info("Collecting SDNET cracked images...")
        for category in ["Decks", "Pavements", "Walls"]:
            cracked_dir = self.config.SDNET_PATH / category / "Cracked"
            if cracked_dir.exists():
                images = self._get_images_from_dir(cracked_dir)
                crack_images.extend(images)
                logger.info(f"  Found {len(images)} crack images in {category}")

        # Из SDNET non-cracked
        logger.info("Collecting SDNET non-cracked images...")
        for category in ["Decks", "Pavements", "Walls"]:
            non_cracked_dir = self.config.SDNET_PATH / category / "Non-cracked"
            if non_cracked_dir.exists():
                images = self._get_images_from_dir(non_cracked_dir)
                non_crack_images.extend(images)
                logger.info(f"  Found {len(images)} non-crack images in {category}")

        logger.info(f"\nTotal crack images: {len(crack_images)}")
        logger.info(f"Total non-crack images: {len(non_crack_images)}")

        # Балансируем датасет
        min_count = min(len(crack_images), len(non_crack_images))
        crack_images = random.sample(crack_images, min_count)
        non_crack_images = random.sample(non_crack_images, min_count)

        logger.info(f"Balanced dataset: {min_count} crack, {min_count} non-crack")

        # Делим на train/val
        self._split_and_copy(crack_images, 'crack')
        self._split_and_copy(non_crack_images, 'no_crack')

        logger.info("YOLO dataset created successfully!")

    def _get_images_from_dir(self, directory):
        """Получает список изображений из директории"""
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        images = []
        for ext in exts:
            images.extend(sorted(Path(directory).glob(ext)))
        return images

    def _split_and_copy(self, image_paths, class_name):
        """Делит изображения на train/val и копирует их"""
        logger.info(f"Processing class: {class_name}")

        split_idx = int(len(image_paths) * self.config.TRAIN_SPLIT)
        train_images = image_paths[:split_idx]
        val_images = image_paths[split_idx:]

        train_dir = self.config.DATASET_ROOT / 'train' / class_name
        val_dir = self.config.DATASET_ROOT / 'val' / class_name

        logger.info(f"  Train: {len(train_images)} -> {train_dir}")
        logger.info(f"  Val: {len(val_images)} -> {val_dir}")

        # Копируем обучающие данные
        for img_path in tqdm(train_images, desc=f"Train {class_name}", leave=False):
            dest = train_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

        # Копируем валидационные данные
        for img_path in tqdm(val_images, desc=f"Val {class_name}", leave=False):
            dest = val_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

    def get_dataset_info(self):
        """Выводит информацию о датасете"""
        logger.info("\n=== YOLO Dataset Info ===")

        train_crack = len(list((self.config.DATASET_ROOT / 'train' / 'crack').glob('*')))
        train_no_crack = len(list((self.config.DATASET_ROOT / 'train' / 'no_crack').glob('*')))
        val_crack = len(list((self.config.DATASET_ROOT / 'val' / 'crack').glob('*')))
        val_no_crack = len(list((self.config.DATASET_ROOT / 'val' / 'no_crack').glob('*')))

        logger.info(f"Train set:")
        logger.info(f"  Crack: {train_crack}")
        logger.info(f"  No crack: {train_no_crack}")
        logger.info(f"  Total: {train_crack + train_no_crack}")

        logger.info(f"Val set:")
        logger.info(f"  Crack: {val_crack}")
        logger.info(f"  No crack: {val_no_crack}")
        logger.info(f"  Total: {val_crack + val_no_crack}")

        logger.info(f"Total dataset size: {train_crack + train_no_crack + val_crack + val_no_crack}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    config = Config()
    converter = DatasetConverter(config)
    converter.build_yolo_dataset()
    converter.get_dataset_info()

if __name__ == "__main__":
    main()

