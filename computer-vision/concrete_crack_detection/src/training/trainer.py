import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from src.utils.config import Config
    from src.data_loaders.unified_loader import UnifiedCrackDataset
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
    from src.utils.metrics import calculate_metrics
    from src.utils.losses import CombinedLoss
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

class Trainer:
    def __init__(self, task='segmentation', save_dir=None, unet_model=None):
        self.config = Config()
        self.task = task
        self.device = self.config.DEVICE
        self.unet_model = unet_model

        self.save_path = Path(save_dir) if save_dir else Path(self.config.BASE_PATH) / "models"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.model_filename = (
            self.config.WEIGHT_NAME_UNET if self.task == 'segmentation'
            else self.config.WEIGHT_NAME_CNN
        )

        dataset = UnifiedCrackDataset(
            crack_dir=self.config.DEEPCRACK_PATH,
            non_crack_dir=self.config.SDNET_PATH,
            task=self.task,
            unet_model=self.unet_model
        )

        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=getattr(self.config, 'NUM_WORKERS', 0)
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            num_workers=getattr(self.config, 'NUM_WORKERS', 0)
        )

        if task == 'segmentation':
            self.model = UNet(n_channels=3, n_classes=1).to(self.device)
            self.criterion = CombinedLoss()
        else:
            self.model = SimpleCNN(n_channels=3, n_classes=1).to(self.device)
            pw = torch.tensor([self.config.POS_WEIGHT_CNN]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def train(self):
        history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

        best_val_loss = float('inf')
        bad_epochs = 0
        early_stopping_limit = self.config.EARLY_STOPPING

        logger.info(f"Starting {self.task} training. Target file: {self.model_filename}")

        for epoch in range(self.config.EPOCHS):
            t_loss, t_met = self.run_epoch(self.train_loader, epoch, is_train=True)
            v_loss, v_met = self.run_epoch(self.val_loader, epoch, is_train=False)

            self.scheduler.step(v_loss)

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_metric'].append(t_met)
            history['val_metric'].append(v_met)

            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch + 1}: LR={current_lr:.6f}, Val Loss={v_loss:.4f}, Metric={v_met:.4f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                bad_epochs = 0
                torch.save(self.model.state_dict(), self.save_path / self.model_filename)
                logger.info(f"Saved new best model to {self.model_filename}")
            else:
                bad_epochs += 1
                if bad_epochs >= early_stopping_limit:
                    logger.warning(f"Early stopping at epoch {epoch + 1}. No improvement for {bad_epochs} epochs.")
                    break

        final_weight_path = self.save_path / self.model_filename
        if final_weight_path.exists():
            self.model.load_state_dict(torch.load(final_weight_path))
            logger.info(f"Training finished. Model loaded with weights from {final_weight_path}")

        return self.model, history

    def run_epoch(self, loader, epoch, is_train):
        self.model.train() if is_train else self.model.eval()
        running_loss = 0.0
        running_metric = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [{'Train' if is_train else 'Val'}]")

        for imgs, masks, labels in pbar:
            imgs, masks, labels = imgs.to(self.device), masks.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = self.model(imgs)
                target = masks if self.task == 'segmentation' else labels.unsqueeze(1)
                loss = self.criterion(outputs, target)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item()
            running_metric += self._calculate_metric(outputs, target)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return running_loss / len(loader), running_metric / len(loader)

    def _calculate_metric(self, pred, target):
        with torch.no_grad():
            if self.task == 'segmentation':
                iou, _ = calculate_metrics(pred, target)
                return iou
            else:
                pred_labels = (torch.sigmoid(pred) > 0.5).float()
                return (pred_labels == target).float().mean().item()