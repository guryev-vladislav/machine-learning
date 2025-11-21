# src/training/segmentation_trainer.py
import torch
import logging
from .base_trainer import BaseTrainer
from ..utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, config)
        self.train_metrics_calculator = SegmentationMetrics()
        self.val_metrics_calculator = SegmentationMetrics()

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        self.train_metrics_calculator.reset()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            self.train_metrics_calculator.update(output, target)

            if batch_idx % 20 == 0:
                logger.info(f'Train batch {batch_idx}, Loss: {loss.item():.6f}')

        return epoch_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        self.val_metrics_calculator.reset()

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                self.val_metrics_calculator.update(output, target)

        return val_loss / len(self.val_loader)

    def train(self, epochs, model_path, result_path):
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            train_metrics = self.train_metrics_calculator.get_all_metrics()
            val_metrics = self.val_metrics_calculator.get_all_metrics()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)

            logger.info(f'Epoch {epoch + 1}/{epochs}')
            logger.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            logger.info(f'Val IoU: {val_metrics["iou"]:.4f}, Val Dice: {val_metrics["dice"]:.4f}')

            if val_metrics['dice'] > self.best_metric:
                self.best_metric = val_metrics['dice']
                self.save_checkpoint(model_path, epoch, val_metrics['dice'])

        self.save_training_data(result_path)
        logger.info("Training completed")