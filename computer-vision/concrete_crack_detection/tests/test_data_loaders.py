import sys
from pathlib import Path
import logging
import torch
import numpy as np

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

# Импортируем загрузчики (предполагаем, что они находятся в src/data_loaders)
try:
    from data_loaders.deepcrack_loader import DeepCrackDataset
    from data_loaders.sdnet_loader import SDNETDataset
except ImportError as e:
    print(f"❌ Ошибка импорта: Не удалось найти модули DeepCrackDataset или SDNETDataset. Проверьте пути в sys.path.")
    print(e)
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ ПУТЕЙ ---
# !!! ВНИМАНИЕ: Замените эти пути на фактические пути к вашим данным !!!
DEEPCACK_ROOT = Path("/home/guryev/Git/machine-learning/computer-vision/concrete_crack_detection/data/external/deepcrack")
SDNET_ROOT = Path("/home/guryev/Git/machine-learning/computer-vision/concrete_crack_detection/data/external/sdnet2018")


# --- КОНФИГУРАЦИЯ ПУТЕЙ ---


def test_deepcrack(data_path):
    logger.info(f"\n--- Тестирование DeepCrack (Сегментация) ---")
    if not data_path.exists():
        logger.error(f"❌ Директория не найдена: {data_path}")
        return

    try:
        # Создаем обучающий набор
        train_dataset = DeepCrackDataset(data_path, is_train=True)
        test_dataset = DeepCrackDataset(data_path, is_train=False)

        logger.info(f"Путь: {data_path.name}")
        logger.info(f"Набор Train: {len(train_dataset)} образцов")
        logger.info(f"Набор Test: {len(test_dataset)} образцов")

        # Тестируем первый элемент
        image, mask = train_dataset[0]

        # Проверяем типы и размеры
        assert isinstance(image, np.ndarray), "Image must be NumPy array"
        assert isinstance(mask, np.ndarray), "Mask must be NumPy array"
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be (H, W, 3)"
        assert mask.ndim == 2, "Mask must be (H, W)"
        assert image.shape[:2] == mask.shape, "Image and Mask shapes must match"

        logger.info(f"✅ DeepCrack - Успех. Image Shape {image.shape}, Mask Shape {mask.shape}")

    except Exception as e:
        logger.error(f"❌ DeepCrack - Ошибка при загрузке или проверке: {e}")


def test_sdnet(data_path):
    logger.info(f"\n--- Тестирование SDNET2018 (Классификация) ---")
    if not data_path.exists():
        logger.error(f"❌ Директория не найдена: {data_path}")
        return

    try:
        # Создаем полный набор
        full_dataset = SDNETDataset(data_path)

        logger.info(f"Путь: {data_path.name}")
        logger.info(f"Общий размер набора: {len(full_dataset)} образцов")

        # Тестируем первый элемент (должен быть с трещиной - label 1)
        image_c, label_c = full_dataset[0]
        # Тестируем последний элемент (должен быть без трещины - label 0, если данные отсортированы)
        image_u, label_u = full_dataset[-1]

        # Проверяем типы и размеры
        assert isinstance(image_c, np.ndarray), "Image must be NumPy array"
        assert isinstance(label_c, (int, float, np.int64, np.float32)), "Label must be scalar"
        assert image_c.ndim == 3 and image_c.shape[2] == 3, "Image must be (H, W, 3)"

        logger.info(f"✅ SDNET - Успех. Image Shape {image_c.shape}, Label (Cracked) {label_c}")
        logger.info(f"✅ SDNET - Успех. Image Shape {image_u.shape}, Label (Uncracked) {label_u}")

    except Exception as e:
        logger.error(f"❌ SDNET - Ошибка при загрузке или проверке: {e}")


if __name__ == "__main__":
    test_deepcrack(DEEPCACK_ROOT)
    test_sdnet(SDNET_ROOT)