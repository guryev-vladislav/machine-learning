# tests/test_unet.py
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
    from src.models.unet import UNet
except ImportError as e:
    pytest.fail(f"Ошибка: Невозможно импортировать UNet. Проверьте путь и файл models/unet.py. Ошибка: {e}")

# --- Параметры для теста ---
BATCH_SIZE = 4
IMAGE_SIZE = 256  # Стандартный размер патча U-Net
N_CHANNELS = 3
N_CLASSES = 1


def test_unet_initialization():
    """Проверяет, что U-Net инициализируется без ошибок."""
    try:
        model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
        assert isinstance(model, UNet), "Объект не является экземпляром UNet."
        print("\nUNet успешно инициализирована.")
    except Exception as e:
        pytest.fail(f"Ошибка инициализации UNet: {e}")


def test_unet_forward_pass():
    """
    Проверяет, что прямой проход через U-Net работает и возвращает
    тензор правильной формы.
    """
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).eval()

    # 1. Создаем фиктивный входной тензор
    # Формат: (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    input_tensor = torch.randn(BATCH_SIZE, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # 2. Выполняем прямой проход
    with torch.no_grad():
        output_logits = model(input_tensor)

    # 3. Проверка формы выходного тензора
    # Должен вернуться тензор той же высоты и ширины, но с N_CLASSES каналами.
    expected_shape = (BATCH_SIZE, N_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    assert output_logits.shape == expected_shape, \
        (f"Неверная форма выходного тензора. Ожидалось: {expected_shape}, "
         f"получено: {tuple(output_logits.shape)}")

    print(f"Прямой проход U-Net успешен. Выходная форма: {tuple(output_logits.shape)}")


def test_unet_parameters_count():
    """Проверяет, что количество параметров модели соответствует ожидаемому (грубая оценка)."""
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    # Считаем общее количество обучаемых параметров
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Стандартная U-Net имеет около 7.7 млн. параметров.
    # Дадим диапазон для гибкости.
    MIN_PARAMS = 7_000_000
    MAX_PARAMS = 8_000_000

    assert MIN_PARAMS < num_params < MAX_PARAMS, \
        (f"Количество параметров ({num_params}) не в ожидаемом диапазоне "
         f"({MIN_PARAMS} - {MAX_PARAMS}). Возможно, в архитектуре есть ошибка.")

    print(f"Количество параметров U-Net: {num_params}.")

# --- Конец test_unet.py ---