import pytest
import sys
import os
from pathlib import Path
import numpy as np

# --- 1. Настройка путей для импорта ---
# Добавляем корневую директорию проекта в путь для импорта модулей,
# что позволяет импортировать 'src.data_loaders.deepcrack_loader'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импорт загрузчиков с префиксом 'src'
try:
    from src.data_loaders.deepcrack_loader import DeepCrackDataset
    from src.data_loaders.sdnet_loader import SDNETDataset
except ImportError as e:
    pytest.fail(f"Ошибка импорта: Не удалось найти модули DeepCrackDataset или SDNETDataset. {e}")


# --- 2. ФИКСТУРЫ ДЛЯ ПУТЕЙ ---

# Используйте эти фикстуры для инъекции путей в тесты
@pytest.fixture(scope="session")
def deepcrack_data_path():
    """Возвращает путь к корневой папке датасета DeepCrack."""
    return Path("/home/guryev/Git/machine-learning/computer-vision/concrete_crack_detection/data/external/deepcrack")


@pytest.fixture(scope="session")
def sdnet_data_path():
    """Возвращает путь к корневой папке датасета SDNET2018."""
    return Path("/home/guryev/Git/machine-learning/computer-vision/concrete_crack_detection/data/external/sdnet2018")


# --- 3. ТЕСТЫ ---

def test_deepcrack_initialization(deepcrack_data_path):
    """Проверяет инициализацию и получение элемента из DeepCrackDataset (задача Сегментации)."""

    if not deepcrack_data_path.exists():
        pytest.skip(f"Директория DeepCrack не найдена: {deepcrack_data_path}. Пропуск теста.")

    # Предполагаем train_ratio=0.8, как в DeepCrackDataset по умолчанию
    try:
        train_dataset = DeepCrackDataset(deepcrack_data_path, is_train=True)
        test_dataset = DeepCrackDataset(deepcrack_data_path, is_train=False)

        assert len(train_dataset) > 0, "Обучающий набор DeepCrack пуст."
        assert len(test_dataset) > 0, "Тестовый набор DeepCrack пуст."

        # Тестируем первый элемент
        image, mask = train_dataset[0]

        # Проверяем типы и размеры
        assert isinstance(image, np.ndarray), "Изображение должно быть массивом NumPy."
        assert isinstance(mask, np.ndarray), "Маска должна быть массивом NumPy."
        assert image.ndim == 3 and image.shape[2] == 3, "Изображение должно иметь форму (H, W, 3)."
        assert mask.ndim == 2, "Маска должна иметь форму (H, W)."
        assert image.shape[:2] == mask.shape, "Формы Изображения и Маски должны совпадать."

    except Exception as e:
        pytest.fail(f"Ошибка при загрузке или проверке DeepCrackDataset: {e}")


def test_sdnet_initialization(sdnet_data_path):
    """Проверяет инициализацию и получение элемента из SDNETDataset (задача Классификации)."""

    if not sdnet_data_path.exists():
        pytest.skip(f"Директория SDNET2018 не найдена: {sdnet_data_path}. Пропуск теста.")

    try:
        # Создаем полный набор (SDNETDataset не использует is_train для сплита)
        full_dataset = SDNETDataset(sdnet_data_path)

        assert len(full_dataset) > 0, "Набор SDNET2018 пуст."

        # Тестируем элементы
        image_c, label_c = full_dataset[0]
        image_u, label_u = full_dataset[-1]

        # Проверяем типы и размеры для первого элемента
        assert isinstance(image_c, np.ndarray), "Изображение должно быть массивом NumPy."
        assert isinstance(label_c, (int, float, np.int64, np.float32)), "Метка должна быть скаляром."
        assert image_c.ndim == 3 and image_c.shape[2] == 3, "Изображение должно иметь форму (H, W, 3)."

        # Проверяем ожидаемые метки
        assert label_c == 1, "Первый элемент должен быть 'Cracked' (метка 1)."
        assert label_u == 0, "Последний элемент должен быть 'Non-cracked' (метка 0)."

    except Exception as e:
        pytest.fail(f"Ошибка при загрузке или проверке SDNETDataset: {e}")