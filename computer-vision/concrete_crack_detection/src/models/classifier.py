import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Вспомогательный Блок Свертки (ConvBlock) ---
class ConvBlock(nn.Module):
    """
    Простой блок, состоящий из свертки, BatchNorm и ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# --- Основная Архитектура SimpleCNN ---
class SimpleCNN(nn.Module):
    """
    Простая сверточная нейронная сеть для бинарной классификации.
    Предназначена для датасета SDNET2018 (трещина/нет трещины).
    """

    def __init__(self, n_channels, n_classes):
        """
        n_channels: 3 (RGB)
        n_classes: 1 (для бинарной классификации с Sigmoid)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 1. Сверточный Блок (Извлечение признаков)
        self.features = nn.Sequential(
            ConvBlock(n_channels, 32),
            nn.MaxPool2d(2, 2),  # 256 -> 128

            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            ConvBlock(256, 256),  # Финальный размер: 256 x 16 x 16
        )

        # 2. Классификатор (Полносвязные слои)
        # Входной размер: 16 * 16 * 256 = 65536

        self.classifier = nn.Sequential(
            # ИСПРАВЛЕНИЕ: Изменено с 512 на 256 для попадания в диапазон 15M-18M
            nn.Linear(256 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Для регуляризации
            nn.Linear(256, n_classes)  # Финальный выход - n_classes (1)
        )

    def forward(self, x):
        # 1. Извлечение признаков
        x = self.features(x)

        # 2. Динамическое определение размера для Flatten
        x = torch.flatten(x, 1)

        # 3. Классификация
        logits = self.classifier(x)

        return logits