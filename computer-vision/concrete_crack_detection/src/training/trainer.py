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
    from src.data_loaders.deepcrack_loader import DeepCrackDataset
    from src.data_loaders.sdnet_loader import SDNETDataset
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_size = (self.config.PATCH_SIZE, self.config.PATCH_SIZE)

        if task == 'segmentation':
            self.model = UNet(n_channels=3, n_classes=1).to(self.device)
            self.dataset = DeepCrackDataset(self.config.DEEPCRACK_PATH, target_size=t_size)
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.model = SimpleCNN(n_channels=3, n_classes=1).to(self.device)
            self.dataset = SDNETDataset(self.config.SDNET_PATH, target_size=t_size)
            p_weight = torch.tensor([self.config.data['training']['pos_weight']]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=p_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.data['training'].get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

    def _calculate_metric(self, pred, target):
        with torch.no_grad():
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            if self.task == 'segmentation':
                intersection = (pred_bin * target).sum()
                union = pred_bin.sum() + target.sum() - intersection
                return (intersection / (union + 1e-7)).item()
            else:
                return (pred_bin == target).float().mean().item()

    def train(self):
        self.model.train()
        logger.info(f"Starting {self.task.upper()} training session")
        history = {'train_loss': [], 'metric': []}

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_metric = 0.0

            pbar = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader),
                        desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

            for i, (images, labels) in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.task == 'segmentation':
                    if labels.dim() == 3: labels = labels.unsqueeze(1)
                else:
                    labels = labels.view(-1, 1)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                current_metric = self._calculate_metric(outputs, labels)

                running_loss += current_loss
                running_metric += current_metric
                pbar.set_postfix(loss=f"{current_loss:.4f}", metric=f"{current_metric:.4f}")

            avg_loss = running_loss / len(self.train_loader)
            avg_metric = running_metric / len(self.train_loader)
            history['train_loss'].append(avg_loss)
            history['metric'].append(avg_metric)
            logger.info(f"Epoch {epoch + 1} summary: Loss={avg_loss:.4f}, Metric={avg_metric:.4f}")

        return history, self.model.state_dict()