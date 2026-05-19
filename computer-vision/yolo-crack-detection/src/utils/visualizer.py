import csv
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
except Exception as e:
    logger.error(f"Visualization dependencies missing: {e}")


def plot_training_history(history, save_path=None, title="YOLOv8 Training History"):
    """
    Визуализирует историю обучения модели
    
    Args:
        history: словарь с ключами 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: путь для сохранения графика
        title: заголовок графика
    """
    if not history:
        logger.warning("No history data to plot")
        return
    
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig)
    
    # Loss plot
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
        plt.close()
    else:
        plt.show()


def _safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def load_ultralytics_history(results_csv):
    """Загружает историю обучения Ultralytics из results.csv (robust к разным именам колонок)."""
    results_csv = Path(results_csv)
    if not results_csv.exists():
        logger.warning(f"results.csv not found: {results_csv}")
        return None

    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if any((v or "").strip() for v in row.values())]

    if not rows:
        logger.warning(f"No rows found in results.csv: {results_csv}")
        return None

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        # unified metric names if available
        "train_metric": [],
        "val_metric": [],
        # fallback legacy names
        "top1": [],
        "top5": [],
    }

    def _get_first_present(row, keys):
        for k in keys:
            if k in row and (row[k] is not None and str(row[k]).strip() != ""):
                return row[k]
        return None

    # candidate keys (ordered) for train/val metric lookup
    train_metric_keys = [
        "metrics/accuracy_top1",
        "train/metrics/accuracy_top1",
        "train/accuracy",
        "train_acc",
        "metrics/top1",
        "metrics/acc",
    ]
    val_metric_keys = [
        "val/metrics/accuracy_top1",
        "metrics/val_accuracy_top1",
        "val/metrics/accuracy",
        "val_accuracy",
        "val_acc",
    ]

    for row in rows:
        epoch = _safe_int(row.get("epoch"))
        if epoch is not None:
            history["epochs"].append(epoch)

        train_loss = _safe_float(row.get("train/loss") or row.get("train_loss"))
        if train_loss is not None:
            history["train_loss"].append(train_loss)

        val_loss = _safe_float(row.get("val/loss") or row.get("val_loss"))
        if val_loss is not None:
            history["val_loss"].append(val_loss)

        # explicit train/val metric extraction
        train_metric_val = _safe_float(_get_first_present(row, train_metric_keys))
        val_metric_val = _safe_float(_get_first_present(row, val_metric_keys))

        if train_metric_val is not None:
            history["train_metric"].append(train_metric_val)

        if val_metric_val is not None:
            history["val_metric"].append(val_metric_val)

        # legacy fallbacks
        top1 = _safe_float(row.get("metrics/accuracy_top1") or row.get("metrics/top1"))
        if top1 is not None:
            history["top1"].append(top1)
            if train_metric_val is None:
                history["train_metric"].append(top1)

        top5 = _safe_float(row.get("metrics/accuracy_top5") or row.get("metrics/top5"))
        if top5 is not None:
            history["top5"].append(top5)

    return history


def plot_ultralytics_training_curves(results_csv, save_path=None, title="YOLO Training Curves"):
    """Строит кривые обучения YOLO из Ultralytics results.csv.
    Если присутствуют per-epoch train/val метрики — рисует их (Train vs Val).
    Иначе — откатывается к Top-1 / Top-5 (совместимость).
    """
    history = load_ultralytics_history(results_csv)
    if not history:
        return

    epochs = history.get("epochs") or list(range(1, max(
        len(history.get("train_loss", [])),
        len(history.get("val_loss", [])),
        len(history.get("train_metric", [])),
        len(history.get("val_metric", [])),
        len(history.get("top1", [])),
        len(history.get("top5", [])),
    ) + 1))

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    if history.get("train_loss"):
        ax1.plot(epochs[:len(history["train_loss"])], history["train_loss"], label="Train", linewidth=2)
    if history.get("val_loss"):
        ax1.plot(epochs[:len(history["val_loss"])], history["val_loss"], label="Val", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("YOLO: Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])

    # Prefer plotting train_metric vs val_metric if available
    has_train_val_metric = bool(history.get("train_metric")) or bool(history.get("val_metric"))
    if has_train_val_metric:
        if history.get("train_metric"):
            ax2.plot(epochs[:len(history["train_metric"])], history["train_metric"], label="Train", linewidth=2)
        if history.get("val_metric"):
            ax2.plot(epochs[:len(history["val_metric"])], history["val_metric"], label="Val", linewidth=2)
        ax2.set_ylabel("Metric", fontsize=12)
        ax2.set_title("YOLO: Metric (Train vs Val)", fontsize=14, fontweight="bold")
    else:
        # Fallback to Top-1 / Top-5
        if history.get("top1"):
            ax2.plot(epochs[:len(history["top1"])], history["top1"], label="Top-1", linewidth=2)
        if history.get("top5"):
            ax2.plot(epochs[:len(history["top5"])], history["top5"], label="Top-5", linewidth=2)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("YOLO: Metric", fontsize=14, fontweight="bold")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved YOLO training curves to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_model_comparison(models_data, save_path=None):
    """
    Сравнивает несколько моделей по различным метрикам
    
    Args:
        models_data: список словарей типа {
            'name': 'Model Name',
            'accuracy': 0.95,
            'f1': 0.92,
            'precision': 0.94,
            'recall': 0.90
        }
        save_path: путь для сохранения графика
    """
    if not models_data:
        logger.warning("No models data to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        names = [m['name'] for m in models_data if metric in m]
        values = [m[metric] for m in models_data if metric in m]
        
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric.capitalize()}', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_detection_results(image_path, detections, save_path=None, confidence_threshold=0.5):
    """
    Визуализирует результаты детектирования на изображении
    
    Args:
        image_path: путь к изображению
        detections: список детектирований (боксов)
        save_path: путь для сохранения результата
        confidence_threshold: порог уверенности для отображения
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # Рисуем боксы
    colors = {'crack': 'red', 'no_crack': 'green'}
    for det in detections:
        if det.get('confidence', 0) < confidence_threshold:
            continue
        
        cls = det.get('class', 'unknown')
        conf = det.get('confidence', 0)
        box = det.get('box', None)  # Expected: (x1, y1, x2, y2)
        
        if box is None:
            continue
        
        x1, y1, x2, y2 = box
        color = colors.get(cls, 'blue')
        
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        label = f"{cls}: {conf:.2f}"
        ax.text(x1, y1 - 5, label, fontsize=10, color=color, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title(f'Detection Results: {Path(image_path).stem}', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved detection results to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_class_distribution(class_counts, save_path=None, title="Class Distribution"):
    """
    Визуализирует распределение классов в датасете
    
    Args:
        class_counts: словарь типа {'crack': 500, 'no_crack': 1000}
        save_path: путь для сохранения графика
        title: заголовок графика
    """
    if not class_counts:
        logger.warning("No class data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.get_cmap('Set2')(np.linspace(0, 1, len(classes)))

    bars = axes[0].bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Absolute Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Pie plot
    axes[1].pie(counts, labels=classes, autopct='%1.1f%%', colors=colors,
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Relative Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_comprehensive_report(metrics_dict, save_path=None, run_metadata=None):
    """
    Создает комплексный отчет о производительности модели
    
    Args:
        metrics_dict: словарь типа {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.92,
            'f1': 0.93,
            'roc_auc': 0.97
        }
        save_path: путь для сохранения графика
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Radar chart
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics_dict.get('accuracy', 0),
        metrics_dict.get('precision', 0),
        metrics_dict.get('recall', 0),
        metrics_dict.get('f1', 0),
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax1.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax1.fill(angles, values, alpha=0.25, color='blue')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Classification Metrics', fontsize=12, fontweight='bold', pad=20)
    ax1.grid(True)
    
    # Metrics summary block
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    metrics_text = (
        f"Accuracy: {metrics_dict.get('accuracy', 0):.4f}\n"
        f"Precision: {metrics_dict.get('precision', 0):.4f}\n"
        f"Recall: {metrics_dict.get('recall', 0):.4f}\n"
        f"F1-Score: {metrics_dict.get('f1', 0):.4f}\n"
        f"ROC AUC: {metrics_dict.get('roc_auc', 0):.4f}\n"
        f"Confusion Matrix: {metrics_dict.get('confusion_matrix', [])}"
    )
    ax2.text(
        0.05,
        0.5,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=12,
        va='center',
        ha='left',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    )
    
    # Bottom plots
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Add summary text
    summary_text = f"""
    Model Performance Summary
    
    • Overall Accuracy: {metrics_dict.get('accuracy', 0):.2%}
    • Precision (True Positives / All Positives): {metrics_dict.get('precision', 0):.2%}
    • Recall (True Positives / Actual Positives): {metrics_dict.get('recall', 0):.2%}
    • F1-Score (Harmonic Mean): {metrics_dict.get('f1', 0):.4f}
    • ROC AUC (Area Under Curve): {metrics_dict.get('roc_auc', 0):.4f}
    
    The model demonstrates {'strong' if metrics_dict.get('f1', 0) > 0.85 else 'moderate'} performance
    in crack detection tasks.
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Comprehensive Model Report', fontsize=16, fontweight='bold')

    if run_metadata:
        meta_lines = []
        for key in ['run_name', 'epochs', 'batch_size', 'imgsz', 'device', 'dataset_root']:
            if key in run_metadata and run_metadata[key] is not None:
                meta_lines.append(f"{key}: {run_metadata[key]}")
        if meta_lines:
            fig.text(
                0.5,
                0.02,
                ' | '.join(meta_lines),
                ha='center',
                va='bottom',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
            )
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comprehensive report to {save_path}")
        plt.close()
    else:
        plt.show()


def save_detection_comparison(original_path, annotated_path, save_path=None):
    """
    Сохраняет сравнение оригинального и аннотированного изображений
    
    Args:
        original_path: путь к оригинальному изображению
        annotated_path: путь к аннотированному изображению
        save_path: путь для сохранения сравнения
    """
    orig = cv2.imread(str(original_path))
    annot = cv2.imread(str(annotated_path))
    
    if orig is None or annot is None:
        logger.error("Failed to load images for comparison")
        return
    
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    annot_rgb = cv2.cvtColor(annot, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(orig_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(annot_rgb)
    axes[1].set_title('Crack Detections', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {save_path}")
        plt.close()
    else:
        plt.show()
