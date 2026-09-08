import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

from src.utils.config import Config
from src.data_preparation.dataset_converter import DatasetConverter
from src.models.yolo_crack_detector import YOLOCrackDetector
from src.utils.visualizer import plot_class_distribution, plot_comprehensive_report

def generate_metrics_report(config, results, detector=None):
    try:
        metrics = {
            'training_date': datetime.now().isoformat(),
            'model': 'YOLOv8n-cls',
            'task': 'Classification',
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
        }

        metrics_file = config.OUTPUTS_PATH / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")

        report_file = config.OUTPUTS_PATH / "training_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLO CRACK DETECTION - TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: YOLOv8n-cls (Nano Classification)\n")
            f.write(f"Task: Binary Classification (crack / no_crack)\n\n")
            f.write("Configuration:\n")
            f.write(f"  Epochs: {config.EPOCHS}\n")
            f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
            f.write(f"  Learning Rate: {config.LR}\n")
            f.write(f"  Device: {config.DEVICE}\n")

        logger.info(f"Training report saved to {report_file}")
        try:
            from src.utils.metrics import classification_metrics
            import cv2
            if detector is None:
                try:
                    detector = YOLOCrackDetector(config=config)
                except Exception:
                    detector = None

            y_true = []
            y_scores = []
            y_pred = []

            val_root = config.DATASET_ROOT / 'val'
            if val_root.exists():
                for idx, cls in enumerate(config.CLASS_NAMES):
                    cls_dir = val_root / cls
                    if not cls_dir.exists():
                        continue
                    for img_path in cls_dir.glob('*'):
                        try:
                            img = cv2.imread(str(img_path))
                            if img is None:
                                continue
                            if detector is None:
                                continue
                            res = detector.classify_frame(img)
                            if res is None:
                                continue
                            y_true.append(1 if cls == 'crack' else 0)
                            y_pred.append(int(res.get('predicted_label', 1 if res.get('is_crack') else 0)))
                            y_scores.append(float(res.get('crack_probability', res.get('confidence', 0.0))))
                        except Exception as e:
                            logger.debug(f"Skipping {img_path} during eval: {e}")

            if y_true and y_scores:
                cls_metrics = classification_metrics(y_true, y_scores=y_scores, y_pred=y_pred, threshold=config.CONF_THRESHOLD)
                metrics.update({'classification': cls_metrics})
                try:
                    with open(config.OUTPUTS_PATH / 'classification_metrics.json', 'w') as f:
                        json.dump(cls_metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not save classification metrics JSON: {e}")
                try:
                    plot_comprehensive_report(
                        cls_metrics,
                        save_path=config.OUTPUTS_PATH / 'training_metrics.png',
                        run_metadata={
                            'run_name': config.OUTPUTS_PATH.name,
                            'epochs': config.EPOCHS,
                            'batch_size': config.BATCH_SIZE,
                            'imgsz': config.IMGSZ,
                            'device': config.DEVICE,
                            'dataset_root': str(config.DATASET_ROOT),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not save comprehensive plot: {e}")

        except Exception as e:
            logger.warning(f"Could not run validation evaluation: {e}")

        return metrics
    except Exception as e:
        logger.warning(f"Could not generate detailed metrics: {e}")
        return {}


def evaluate_weights_over_epochs(config: Config, detector: YOLOCrackDetector, training_dir: Path):
    try:
        from src.utils.metrics import classification_metrics
        import cv2
        import re
        from src.utils.visualizer import plot_training_history, load_ultralytics_history
    except Exception as e:
        logger.warning(f"Could not import evaluation dependencies: {e}")
        return

    weights_dir = Path(training_dir) / 'weights'
    if not weights_dir.exists():
        logger.info(f"No weights directory found at {weights_dir}")
        return
    candidates = [p for p in weights_dir.glob('*.pt') if p.name not in ('best.pt', 'last.pt')]
    if not candidates:
        candidates = [p for p in weights_dir.iterdir() if p.suffix == '.pt' and p.name not in ('best.pt','last.pt')]
    if not candidates:
        logger.info("No per-epoch weight files found (only best.pt/last.pt). Skipping per-epoch eval.")
        return

    def epoch_key(p: Path):
        m = re.search(r"(\d+)", p.name)
        if m:
            return int(m.group(1))
        return int(p.stat().st_mtime)

    candidates = sorted(candidates, key=epoch_key)

    val_root = config.DATASET_ROOT / 'val'
    if not val_root.exists():
        logger.info(f"Val dataset not found at {val_root}. Skipping per-epoch eval.")
        return

    per_epoch_val_acc = []
    epochs = []

    for w in candidates:
        try:
            logger.info(f"Evaluating weights: {w}")
            det = YOLOCrackDetector(config=config, model_path=str(w))
            y_true = []
            y_scores = []
            y_pred = []
            for idx, cls in enumerate(config.CLASS_NAMES):
                cls_dir = val_root / cls
                if not cls_dir.exists():
                    continue
                for img_path in cls_dir.glob('*'):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        res = det.classify_frame(img)
                        if res is None:
                            continue
                        y_true.append(1 if cls == 'crack' else 0)
                        y_pred.append(int(res.get('predicted_label', 1 if res.get('is_crack') else 0)))
                        y_scores.append(float(res.get('crack_probability', res.get('confidence', 0.0))))
                    except Exception as e:
                        logger.debug(f"Skipping {img_path} during epoch eval: {e}")

            if y_true:
                cls_metrics = classification_metrics(y_true, y_scores=y_scores, y_pred=y_pred, threshold=config.CONF_THRESHOLD)
                per_epoch_val_acc.append(cls_metrics.get('accuracy', 0.0))
                m = re.search(r"(\d+)", w.name)
                if m:
                    epochs.append(int(m.group(1)))
                else:
                    epochs.append(len(epochs) + 1)
            else:
                per_epoch_val_acc.append(0.0)
                epochs.append(len(epochs) + 1)
        except Exception as e:
            logger.warning(f"Failed to evaluate {w}: {e}")

    out = config.OUTPUTS_PATH / 'per_epoch_val_metrics.json'
    try:
        with open(out, 'w') as f:
            json.dump({'epochs': epochs, 'val_accuracy': per_epoch_val_acc}, f, indent=2)
        logger.info(f"Saved per-epoch val metrics to {out}")
    except Exception as e:
        logger.warning(f"Could not save per-epoch metrics JSON: {e}")

    results_csv = Path(training_dir) / 'results.csv'
    history = None
    if results_csv.exists():
        history = load_ultralytics_history(results_csv)

    plot_hist = {
        'train_loss': history.get('train_loss') if history else [],
        'val_loss': history.get('val_loss') if history else [],
        'train_acc': history.get('train_metric') if history else history.get('top1') if history else [],
        'val_acc': per_epoch_val_acc,
    }

    try:
        plot_training_history(plot_hist, save_path=config.OUTPUTS_PATH / 'plots' / 'yolo_curves.png', title='YOLO: Loss & Metric (Train vs Val)')
        logger.info(f"Saved combined training curves to {config.OUTPUTS_PATH / 'plots' / 'yolo_curves.png'}")
    except Exception as e:
        logger.warning(f"Could not plot combined training curves: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO crack detection model')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare dataset')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--all', action='store_true', help='Prepare data and train')

    args = parser.parse_args()

    config = Config()
    config.create_directories()

    if args.prepare_data or args.all:
        logger.info("=" * 60)
        logger.info("Step 1: Preparing dataset...")
        logger.info("=" * 60)

        converter = DatasetConverter(config)
        converter.build_yolo_dataset()
        dataset_info = converter.get_dataset_info()

        try:
            if dataset_info:
                plot_class_distribution(
                    dataset_info,
                    save_path=config.OUTPUTS_PATH / "class_distribution.png",
                    title="Dataset Class Distribution"
                )
                logger.info("Class distribution plot saved")
        except Exception as e:
            logger.warning(f"Could not save class distribution plot: {e}")

    if args.train or args.all:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Training YOLO model...")
        logger.info("=" * 60)

        detector = YOLOCrackDetector(config=config)

        epochs = args.epochs or config.EPOCHS
        batch_size = args.batch_size or config.BATCH_SIZE

        results = detector.train(epochs=epochs, batch_size=batch_size)

        logger.info("\n" + "=" * 60)
        logger.info("Training completed!")
        logger.info("=" * 60)

        metrics = generate_metrics_report(config, results, detector=detector)

        model_path = config.OUTPUTS_PATH / "best_model.pt"
        detector.save_model(model_path)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"All outputs saved to: {config.OUTPUTS_PATH}")
        logger.info("\n Generated files:")
        logger.info(f"  - best_model.pt (trained weights)")
        logger.info(f"  - metrics.json (performance metrics)")
        logger.info(f"  - training_report.txt (human-readable report)")
        if (config.OUTPUTS_PATH / "class_distribution.png").exists():
            logger.info(f"  - class_distribution.png")

if __name__ == "__main__":
    from src.utils.logging_setup import shutdown_logging

    try:
        main()
    finally:
        shutdown_logging()


