import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

def save_multiscale_comparison(results, save_path):
    n_scales = len(results)
    fig, axes = plt.subplots(n_scales, 2, figsize=(12, 5 * n_scales), squeeze=False)

    for i, (name, mask, prob, input_img) in enumerate(results):
        axes[i, 0].imshow(input_img)
        color = 'green' if prob > 0.5 else 'red'
        axes[i, 0].set_title(f"Input Scale: {name}\nCNN Verdict: {prob:.4f}",
                             fontsize=14, fontweight='bold', color=color)
        axes[i, 0].axis('off')

        mask_uint8 = (mask * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(input_img, 0.6, heatmap, 0.4, 0)

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"UNet Segmentation Overlay", fontsize=12)
        axes[i, 1].axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_path, title="Training Stats"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{title}: Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_metric'], label='Train')
    plt.plot(history['val_metric'], label='Val')
    plt.title(f'{title}: Metric')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()