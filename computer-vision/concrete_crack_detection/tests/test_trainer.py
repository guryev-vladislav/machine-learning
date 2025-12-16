# tests/test_trainer.py
import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# --- Настройка путей для импорта ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импорт Trainer и моделей
try:
    from src.training.trainer import Trainer, get_dataloaders
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
    from src.utils.losses import CombinedLoss
    # Импорт классов, которые мы будем мокать
    import src.data_loaders.deepcrack_loader as deepcrack_module
    import src.data_loaders.sdnet_loader as sdnet_module
except ImportError as e:
    pytest.fail(f"Ошибка: Невозможно импортировать модули. Проверьте пути: {e}")

# --- Глобальная Фиктивная Дата (Mock Data) ---
# Создаем фиктивный батч, который будет возвращать наш Mock-DataLoader
MOCK_IMAGE = torch.randn(2, 3, 256, 256)  # (Batch, C, H, W)
MOCK_SEG_MASK = torch.randint(0, 2, (2, 1, 256, 256)).float()  # Сегментация: (Batch, 1, H, W)
MOCK_CLS_LABEL = torch.randint(0, 2, (2,)).float()  # Классификация: (Batch)


# --- Фикстуры для Моков Датасетов ---

@pytest.fixture(autouse=True)
def mock_datasets(monkeypatch):
    """
    Мокируем классы датасетов, чтобы они не обращались к диску.
    """

    # 1. Мок для DeepCrackDataset (Сегментация)
    MockDeepCrackDataset = MagicMock()
    # Задаем __len__ для Datasets
    MockDeepCrackDataset.return_value.__len__.return_value = 10
    # Задаем __getitem__ для Datasets (но в тесте используется DataLoader,
    # который берет item, но нам нужен mock для всего объекта).

    monkeypatch.setattr(deepcrack_module, "DeepCrackDataset", MockDeepCrackDataset)

    # 2. Мок для SDNETDataset (Классификация)
    MockSDNETDataset = MagicMock()
    MockSDNETDataset.return_value.__len__.return_value = 10

    monkeypatch.setattr(sdnet_module, "SDNETDataset", MockSDNETDataset)

    # 3. Мок для get_dataloaders, чтобы он возвращал Mock-DataLoader
    def mock_get_dataloaders(task, data_root, batch_size, image_size):
        # Создаем Mock-DataLoader
        mock_loader = MagicMock()

        if task == 'segmentation':
            # Имитация батча (Image, Mask) для сегментации
            mock_loader.__iter__.return_value = [(MOCK_IMAGE, MOCK_SEG_MASK)]

            # Для проверки len(trainer.train_loader.dataset)
            mock_loader.dataset = MockDeepCrackDataset()
        else:  # classification
            # Имитация батча (Image, Label) для классификации
            mock_loader.__iter__.return_value = [(MOCK_IMAGE, MOCK_CLS_LABEL)]

            # Для проверки len(trainer.train_loader.dataset)
            mock_loader.dataset = MockSDNETDataset()

        # Важно: len DataLoader должен работать для _train_epoch
        mock_loader.__len__.return_value = 1

        return mock_loader, mock_loader

    # Заменяем оригинальную функцию get_dataloaders на наш мок
    monkeypatch.setattr("src.training.trainer.get_dataloaders", mock_get_dataloaders)


# --- Тесты ---

def test_trainer_initialization_segmentation(mock_datasets):
    """Проверяет, что Trainer правильно инициализируется для задачи 'segmentation'."""
    try:
        # Теперь Trainer инициализируется, используя мокированные загрузчики
        trainer = Trainer(task='segmentation', num_epochs=1, batch_size=2, lr=1e-3)

        assert isinstance(trainer.model, UNet), "Для сегментации должна использоваться UNet."
        assert isinstance(trainer.criterion, CombinedLoss), "Для сегментации должна использоваться CombinedLoss."
        assert trainer.task == 'segmentation'
        # Проверяем, что датасет имеет ненулевую длину (благодаря Mock)
        assert len(trainer.train_loader.dataset) > 0, "DataLoader для сегментации пуст."

        print("\nTrainer для сегментации успешно инициализирован.")

    except Exception as e:
        pytest.fail(f"Ошибка инициализации Trainer (segmentation): {e}")


def test_trainer_initialization_classification(mock_datasets):
    """Проверяет, что Trainer правильно инициализируется для задачи 'classification'."""
    try:
        trainer = Trainer(task='classification', num_epochs=1, batch_size=2, lr=1e-3)

        assert isinstance(trainer.model, SimpleCNN), "Для классификации должна использоваться SimpleCNN."
        assert isinstance(trainer.criterion,
                          nn.BCEWithLogitsLoss), "Для классификации должна использоваться BCEWithLogitsLoss."
        assert trainer.task == 'classification'
        assert len(trainer.train_loader.dataset) > 0, "DataLoader для классификации пуст."

        print("Trainer для классификации успешно инициализирован.")

    except Exception as e:
        pytest.fail(f"Ошибка инициализации Trainer (classification): {e}")


def test_trainer_single_training_step(mock_datasets):
    """
    Проверяет, что Trainer может выполнить один тренировочный шаг (без падений)
    для классификации (т.к. она проще).
    """
    trainer = Trainer(task='classification', num_epochs=1, batch_size=2, lr=1e-3)

    try:
        # Выполняем тренировку одной эпохи (благодаря моку DataLoader, это будет 1 итерация)
        avg_loss, metric = trainer._train_epoch()

        # Проверяем, что возвращаемые значения — числа
        assert isinstance(avg_loss, float), "Средняя потеря должна быть float."
        assert isinstance(metric, float), "Метрика должна быть float."

        # Проверяем, что потери и метрика не NaN или Inf (хотя это почти всегда так при рандомном входе)
        assert not torch.isnan(torch.tensor(avg_loss)).item()

        print("\nTrainer успешно выполнил одну тренировочную эпоху.")

    except Exception as e:
        pytest.fail(f"Ошибка при выполнении тренировочного шага: {e}")