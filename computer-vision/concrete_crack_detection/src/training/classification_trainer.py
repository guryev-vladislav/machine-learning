# src/training/classification_trainer.py
import torch
import logging
from .base_trainer import BaseTrainer
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, config)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if batch_idx % 20 == 0:
                logger.info(f'Classification batch {batch_idx}, Loss: {loss.item():.6f}')

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')

        return epoch_loss / len(self.train_loader), accuracy, f1

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')

        return val_loss / len(self.val_loader), accuracy, f1