# src/training/base_trainer.py
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')

        return epoch_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

        return val_loss / len(self.val_loader)

    def save_checkpoint(self, model_path, epoch, metric):
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metric': metric,
                'config': self.config
            }

            torch.save(checkpoint, Path(model_path) / "model.pth")
            logger.info(f"Checkpoint saved for epoch {epoch}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def save_training_data(self, result_path):
        try:
            training_data = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }

            data_file = Path(result_path) / "training_data" / "metrics_epoch.json"
            with open(data_file, 'w') as f:
                json.dump(training_data, f, indent=2)

            logger.info(f"Training data saved to {data_file}")

        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            raise