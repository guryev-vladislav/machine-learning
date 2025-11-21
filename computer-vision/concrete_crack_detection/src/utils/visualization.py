# src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(train_losses) + 1)

        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        train_ious = [m['iou'] for m in train_metrics]
        val_ious = [m['iou'] for m in val_metrics]
        ax2.plot(epochs, train_ious, 'b-', label='Train IoU')
        ax2.plot(epochs, val_ious, 'r-', label='Val IoU')
        ax2.set_title('IoU Score')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True)

        train_dice = [m['dice'] for m in train_metrics]
        val_dice = [m['dice'] for m in val_metrics]
        ax3.plot(epochs, train_dice, 'b-', label='Train Dice')
        ax3.plot(epochs, val_dice, 'r-', label='Val Dice')
        ax3.set_title('Dice Coefficient')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Dice')
        ax3.legend()
        ax3.grid(True)

        train_f1 = [m['f1'] for m in train_metrics]
        val_f1 = [m['f1'] for m in val_metrics]
        ax4.plot(epochs, train_f1, 'b-', label='Train F1')
        ax4.plot(epochs, val_f1, 'r-', label='Val F1')
        ax4.set_title('F1 Score')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('F1')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(Path(save_path) / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Training curves plot saved")

    except Exception as e:
        logger.error(f"Failed to create training curves: {e}")


def save_prediction_samples(images, masks, predictions, save_path, num_samples=5):
    try:
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(min(num_samples, len(images))):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            mask = masks[i].cpu().squeeze().numpy()
            pred = predictions[i].cpu().squeeze().numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Failed to save prediction samples: {e}")