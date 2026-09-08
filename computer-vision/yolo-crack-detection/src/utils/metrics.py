from pathlib import Path
import numpy as np
from typing import Iterable

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    logger.error(f"Visualization dependencies missing: {e}")


def _load_sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            roc_curve,
            precision_recall_curve,
        )
        return {
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'roc_auc_score': roc_auc_score,
            'confusion_matrix': confusion_matrix,
            'roc_curve': roc_curve,
            'precision_recall_curve': precision_recall_curve,
        }
    except Exception as e:
        logger.error(f"Metrics dependencies missing: {e}")
        return None


def classification_metrics(y_true, y_scores=None, y_pred=None, threshold=0.5, pos_label=1):
    sklearn_metrics = _load_sklearn_metrics()
    if sklearn_metrics is None:
        raise ImportError('scikit-learn is required for classification_metrics')

    y_true = np.asarray(y_true)
    y_scores = None if y_scores is None else np.asarray(y_scores)
    if y_pred is None:
        if y_scores is None:
            raise ValueError('Either y_scores or y_pred must be provided')
        y_pred = (y_scores >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        'accuracy': float(sklearn_metrics['accuracy_score'](y_true, y_pred)),
        'precision': float(sklearn_metrics['precision_score'](y_true, y_pred, zero_division=0)),
        'recall': float(sklearn_metrics['recall_score'](y_true, y_pred, zero_division=0)),
        'f1': float(sklearn_metrics['f1_score'](y_true, y_pred, zero_division=0)),
    }

    try:
        if y_scores is None:
            y_scores = y_pred
        metrics['roc_auc'] = float(sklearn_metrics['roc_auc_score'](y_true, y_scores))
    except Exception:
        metrics['roc_auc'] = 0.0

    try:
        cm = sklearn_metrics['confusion_matrix'](y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
    except Exception:
        metrics['confusion_matrix'] = []

    return metrics


def plot_confusion_matrix(cm, class_names=('no_crack', 'crack'), save_path=None):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title('Confusion Matrix')
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()


def plot_roc(y_true, y_scores, save_path=None):
    try:
        sklearn_metrics = _load_sklearn_metrics()
        if sklearn_metrics is None:
            return
        fpr, tpr, _ = sklearn_metrics['roc_curve'](y_true, y_scores)
        auc = sklearn_metrics['roc_auc_score'](y_true, y_scores)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved ROC curve to {save_path}")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Failed to plot ROC: {e}")


def plot_precision_recall(y_true, y_scores, save_path=None):
    try:
        sklearn_metrics = _load_sklearn_metrics()
        if sklearn_metrics is None:
            return
        prec, rec, _ = sklearn_metrics['precision_recall_curve'](y_true, y_scores)

        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label='Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved PR curve to {save_path}")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Failed to plot PR curve: {e}")


def save_metrics_report(metrics: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Saved metrics report to {path}")


def calculate_segmentation_metrics(pred, target, threshold=0.5):
    try:
        import torch
    except Exception as e:
        logger.error(f"Torch is required for segmentation metrics: {e}")
        raise

    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

    return float(iou.item()), float(dice.item())

