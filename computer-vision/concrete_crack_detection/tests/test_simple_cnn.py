# tests/test_simple_cnn.py
import pytest
import torch
import sys
import os
from pathlib import Path

# --- Настройка путей для импорта ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импорт модели
try:
    from src.models.classifier import SimpleCNN
except ImportError as e:
    pytest.fail(f"Ошибка: Невозможно импортировать SimpleCNN. Проверьте путь и файл models/classifier.py. Ошибка: {e}")

# --- Параметры для теста ---
BATCH_SIZE = 8
IMAGE_SIZE = 256  # SDNET2018 часто использует этот размер или 227
N_CHANNELS = 3
N_CLASSES = 1  # Бинарная классификация

# Ожидаемый диапазон параметров:
# SimpleCNN с 4-мя блоками и полносвязными слоями должен иметь ~16.7M параметров
MIN_PARAMS = 15_000_000
MAX_PARAMS = 18_000_000


@pytest.fixture(scope="module")
def simple_cnn_model():
    """Фикстура для создания и возврата модели SimpleCNN."""
    return SimpleCNN(n_channels=N_CHANNELS, n_classes=N_CLASSES).eval()


def test_simple_cnn_initialization(simple_cnn_model):
    """Проверяет, что SimpleCNN инициализируется без ошибок."""
    assert isinstance(simple_cnn_model, SimpleCNN), "Объект не является экземпляром SimpleCNN."
    print("\nSimpleCNN успешно инициализирована.")


def test_simple_cnn_forward_pass(simple_cnn_model):
    """
    Проверяет, что прямой проход через SimpleCNN работает и возвращает
    тензор правильной формы.
    """
    # 1. Создаем фиктивный входной тензор
    # Формат: (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    input_tensor = torch.randn(BATCH_SIZE, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # 2. Выполняем прямой проход
    with torch.no_grad():
        output_logits = simple_cnn_model(input_tensor)

    # 3. Проверка формы выходного тензора
    # Для классификации ожидается форма (BATCH_SIZE, N_CLASSES)
    expected_shape = (BATCH_SIZE, N_CLASSES)

    assert output_logits.shape == expected_shape, \
        (f"Неверная форма выходного тензора. Ожидалось: {expected_shape}, "
         f"получено: {tuple(output_logits.shape)}")

    print(f"Прямой проход SimpleCNN успешен. Выходная форма: {tuple(output_logits.shape)}")


def test_simple_cnn_parameters_count(simple_cnn_model):
    """Проверяет, что количество параметров модели соответствует ожидаемому диапазону."""
    # Считаем общее количество обучаемых параметров
    num_params = sum(p.numel() for p in simple_cnn_model.parameters() if p.requires_grad)

    assert MIN_PARAMS < num_params < MAX_PARAMS, \
        (f"Количество параметров ({num_params}) не в ожидаемом диапазоне "
         f"({MIN_PARAMS} - {MAX_PARAMS}). Возможно, в архитектуре есть ошибка.")

    print(f"Количество параметров ({num_params}) в ожидаемом диапазоне.")