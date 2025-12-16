# src/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
import sys
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- БЛОК ИМПОРТА: Добавление корневой директории в путь ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# *** ИСПРАВЛЕНИЕ: Добавлен префикс 'src.' ко всем внутренним импортам для унификации путей ***
try:
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
    from src.data_loaders.deepcrack_loader import DeepCrackDataset
    from src.data_loaders.sdnet_loader import SDNETDataset
    from src.utils.losses import CombinedLoss, dice_loss
except ImportError as e:
    logger.error(f"Ошибка импорта модуля: {e}. Проверьте, что все пути начинаются с 'src.'.")
    # В тестовом окружении это может быть допустимо, но в боевом - нет
    sys.exit(1)

# --- ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Ваши пути, которые будут переданы в Trainer
DEEPCACK_ROOT = Path(
    "/home/guryev/Git/machine-learning/computer-vision/concrete_crack_detection/data/external/deepcrack")
SDNET_ROOT = Path("/home/guryev/Git/machine-learning/computer_vision/concrete_crack_detection/data/external/sdnet2018")


# --- 1. Вспомогательные функции (Трансформации) ---

def get_transforms(task, image_size):
    """Возвращает набор трансформаций для тренировки и валидации в зависимости от задачи."""
    if task == 'segmentation':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], is_check_shapes=False)
    elif task == 'classification':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], is_check_shapes=False)
    else:
        raise ValueError(f"Неизвестная задача: {task}")


# --- 2. Функция создания DataLoader'ов ---

def get_dataloaders(task, data_root, batch_size, image_size):
    """Создает DataLoader'ы для заданной задачи."""

    transforms = get_transforms(task, image_size)

    if task == 'segmentation':
        DatasetClass = DeepCrackDataset
    elif task == 'classification':
        DatasetClass = SDNETDataset
    else:
        raise ValueError(f"Неизвестная задача: {task}")

    # Здесь мы используем placeholder для имитации train/val сплита
    # В реальном коде DatasetClass должен сам обрабатывать сплит
    # ПРИМЕЧАНИЕ: В тесте эта функция мокается, чтобы избежать чтения диска
    train_dataset = DatasetClass(data_root, transform=transforms, is_train=True)
    val_dataset = DatasetClass(data_root, transform=transforms, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        f"--- Датасет {DatasetClass.__name__} загружен. Train: {len(train_dataset)}, Val: {len(val_dataset)} ---")

    return train_loader, val_loader


# --- 3. Класс Тренера ---

class Trainer:
    def __init__(self, task: str, num_epochs: int = 10, batch_size: int = 8, lr: float = 1e-3):
        self.task = task
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = DEVICE

        # 1. Настройка модели и критерия
        if task == 'segmentation':
            self.model = UNet(n_channels=3, n_classes=1).to(self.device)
            self.criterion = CombinedLoss().to(self.device)
            data_root = DEEPCACK_ROOT
            self.image_size = 256
        elif task == 'classification':
            # Модель SimpleCNN
            self.model = SimpleCNN(n_channels=3, n_classes=1).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
            data_root = SDNET_ROOT
            self.image_size = 256
        else:
            raise ValueError("Недопустимое значение параметра task. Используйте 'segmentation' или 'classification'.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 2. Инициализация загрузчиков
        # В рабочем режиме это вызывает реальные загрузчики, в тесте - моки
        self.train_loader, self.val_loader = get_dataloaders(self.task, data_root, self.batch_size, self.image_size)

        logger.info(f"Тренер инициализирован. Модель: {self.model.__class__.__name__}, Задача: {self.task}")

    def _train_epoch(self):
        """Логика одной тренировочной эпохи."""
        self.model.train()
        total_loss = 0.0

        # Переменные для метрик
        correct_predictions = 0
        total_samples = 0
        total_dice_score = 0

        # В тесте train_loader содержит всего один фиктивный батч (1 итерация)
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)

            # Различная обработка targets для разных задач
            if self.task == 'segmentation':
                # Сегментация: размер (N, 1, H, W), float
                targets = targets.to(self.device).float()
            elif self.task == 'classification':
                # Классификация: размер (N, 1) - для BCEWithLogitsLoss, float
                targets = targets.to(self.device).float().unsqueeze(1)

            predictions = self.model(data)
            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Расчет метрик
            if self.task == 'segmentation':
                # Для Dice Score используем Sigmoid-активацию
                predicted_probs = torch.sigmoid(predictions)
                # 1 - dice_loss дает нам Dice Score (метрология)
                dice_s = 1 - dice_loss(predicted_probs, targets).item()
                total_dice_score += dice_s

            elif self.task == 'classification':
                # Для Accuracy используем Sigmoid + порог 0.5
                predicted_probs = torch.sigmoid(predictions)
                predicted_classes = (predicted_probs > 0.5).float()
                correct_predictions += (predicted_classes == targets).sum().item()
                total_samples += targets.size(0)

        # Возвращаем средние значения потерь и метрик для эпохи
        avg_loss = total_loss / len(self.train_loader)
        if self.task == 'segmentation':
            metric = total_dice_score / len(self.train_loader)
            metric_name = "Dice Score"
        else:
            metric = correct_predictions / total_samples
            metric_name = "Accuracy"

        return avg_loss, metric

    # Метод валидации, который пригодится вам для полного цикла
    # def _validate_epoch(self):
    #     ...

    def train(self):
        """Основной метод для запуска тренировки."""
        logger.info(f"Запуск тренировки задачи '{self.task}' на {self.num_epochs} эпох...")

        for epoch in range(self.num_epochs):
            train_loss, train_metric = self._train_epoch()

            # Логирование результатов эпохи
            metric_name = "Dice Score" if self.task == 'segmentation' else "Accuracy"
            logger.info(
                f"--- Эпоха {epoch + 1}/{self.num_epochs} завершена. "
                f"Loss: {train_loss:.4f}, {metric_name}: {train_metric:.4f} ---"
            )

        logger.info(f"Тренировка задачи {self.task} завершена.")
        return self.model.state_dict()  # Возвращаем веса