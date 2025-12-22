import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Настройка путей
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Безопасные импорты
try:
    from src.utils.config import Config
    from src.data_loaders.unified_loader import UnifiedCrackDataset
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
    from src.utils.metrics import calculate_metrics
    from src.utils.losses import CombinedLoss  # Импортируем наш комбинированный лосс
except ImportError as e:
    logger.error(f"Error importing in trainer.py: {e}")
    sys.exit(1)


class Trainer:
    def __init__(self, task='segmentation', num_epochs=None, batch_size=None, lr=None):
        self.config = Config()
        self.task = task
        self.num_epochs = num_epochs or self.config.EPOCHS
        self.batch_size = batch_size or self.config.BATCH_SIZE
        self.lr = lr or self.config.LR
        self.device = self.config.DEVICE

        t_size = (self.config.PATCH_SIZE, self.config.PATCH_SIZE)

        self.dataset = UnifiedCrackDataset(
            crack_dir=self.config.DEEPCRACK_PATH,
            non_crack_dir=self.config.SDNET_PATH,
            is_train=True,
            target_size=t_size,
            task=self.task  # <-- Передаем текущую задачу (segmentation или classification)
        )

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if "cuda" in str(self.device) else False
        )

        # Внутри __init__ класса Trainer
        if task == 'segmentation':
            self.model = UNet(n_channels=3, n_classes=1).to(self.device)
            self.criterion = CombinedLoss(weight=2.0)  # Dice имеет двойной вес
        else:
            self.model = SimpleCNN(n_channels=3, n_classes=1).to(self.device)
            # Классификатору тоже даем большой вес на трещины
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss, running_metric = 0.0, 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [{self.task}]")

        for images, masks, labels in pbar:
            images = images.to(self.device)

            if self.task == 'segmentation':
                targets = masks.to(self.device)
            else:
                targets = labels.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_metric += self._calculate_metric(outputs, targets)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader), running_metric / len(self.train_loader)

    def _calculate_metric(self, pred, target):
        with torch.no_grad():
            if self.task == 'segmentation':
                iou, _ = calculate_metrics(pred, target)
                return iou
            else:
                pred_labels = (torch.sigmoid(pred) > 0.5).float()
                return (pred_labels == target).float().mean().item()

    def train(self):
        history = {'train_loss': [], 'metric': []}
        for epoch in range(self.num_epochs):
            loss, metric = self.train_epoch(epoch)
            history['train_loss'].append(loss)
            history['metric'].append(metric)
            logger.info(f"Epoch {epoch + 1}: Loss={loss:.4f}, Metric={metric:.4f}")
        return history

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)