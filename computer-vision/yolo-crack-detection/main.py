import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from src.utils.config import Config
from src.data_preparation.dataset_converter import DatasetConverter
from src.models.yolo_crack_detector import YOLOCrackDetector
from src.utils.visualizer import plot_class_distribution
from train import generate_metrics_report


def _build_run_config(args):
    config = Config()

    if args.device:
        config.DEVICE = args.device
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.imgsz is not None:
        config.IMGSZ = args.imgsz
    if args.dataset_root:
        config.DATASET_ROOT = Path(args.dataset_root)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_name:
        run_name = f'{args.run_name}_{timestamp}'
    else:
        run_name = f'run_{timestamp}'
    config.OUTPUTS_PATH = config.OUTPUTS_PATH / run_name
    config.create_directories()

    return config, run_name, timestamp


def _copy_tree(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.rglob('*'):
        if path.is_dir():
            continue
        rel = path.relative_to(source_dir)
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _resolve_training_dir(detector, results):
    candidates = []
    if results is not None:
        candidates.append(getattr(results, 'save_dir', None))
    trainer = getattr(getattr(detector, 'model', None), 'trainer', None)
    if trainer is not None:
        candidates.append(getattr(trainer, 'save_dir', None))

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _write_run_manifest(config, run_name, args, metrics=None, training_dir=None):
    manifest = {
        'run_name': run_name,
        'training_date': datetime.now().isoformat(),
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'imgsz': config.IMGSZ,
        'device': config.DEVICE,
        'dataset_root': str(config.DATASET_ROOT),
        'outputs_path': str(config.OUTPUTS_PATH),
        'training_dir': str(training_dir) if training_dir else None,
        'args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'imgsz': args.imgsz,
            'device': args.device,
            'dataset_root': args.dataset_root,
            'run_name': args.run_name,
        },
    }
    if metrics:
        manifest['metrics'] = metrics

    manifest_path = config.OUTPUTS_PATH / 'run_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    parser = argparse.ArgumentParser(description='Run the full YOLO crack detection pipeline')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None, help='Training image size')
    parser.add_argument('--device', type=str, default=None, help='Training device, e.g. cpu or cuda:0')
    parser.add_argument('--dataset-root', type=str, default=None, help='Override dataset root path')
    parser.add_argument('--run-name', type=str, default=None, help='Optional custom run folder name')
    parser.add_argument('--skip-prepare-data', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    args = parser.parse_args()

    config, run_name, _timestamp = _build_run_config(args)
    logger.info(f'Run folder: {config.OUTPUTS_PATH}')

    _write_run_manifest(config, run_name, args)

    if not args.skip_prepare_data:
        logger.info('=' * 60)
        logger.info('Step 1: Preparing dataset...')
        logger.info('=' * 60)

        converter = DatasetConverter(config)
        converter.build_yolo_dataset()
        dataset_info = converter.get_dataset_info()

        try:
            if dataset_info:
                plot_class_distribution(
                    dataset_info,
                    save_path=config.OUTPUTS_PATH / 'class_distribution.png',
                    title='Dataset Class Distribution'
                )
                logger.info('✓ Class distribution plot saved')
        except Exception as e:
            logger.warning(f'Could not save class distribution plot: {e}')

    if args.skip_train:
        logger.info('Training skipped by flag. Only dataset preparation was executed.')
        return

    logger.info('\n' + '=' * 60)
    logger.info('Step 2: Training YOLO model...')
    logger.info('=' * 60)

    detector = YOLOCrackDetector(config=config)
    results = detector.train(dataset_root=config.DATASET_ROOT, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

    logger.info('\n' + '=' * 60)
    logger.info('Training completed!')
    logger.info('=' * 60)

    metrics = generate_metrics_report(config, results, detector=detector)

    training_dir = _resolve_training_dir(detector, results)
    if training_dir:
        _copy_tree(training_dir, config.OUTPUTS_PATH / 'ultralytics_run')
        best_pt = training_dir / 'weights' / 'best.pt'
        if best_pt.exists():
            shutil.copy2(best_pt, config.OUTPUTS_PATH / 'best_model.pt')
            logger.info(f'✓ Best model copied to: {config.OUTPUTS_PATH / "best_model.pt"}')

    # Fallback: if best.pt wasn't found, still save the in-memory model
    if not (config.OUTPUTS_PATH / 'best_model.pt').exists():
        try:
            detector.save_model(config.OUTPUTS_PATH / 'best_model.pt')
            logger.info(f'✓ Model saved to: {config.OUTPUTS_PATH / "best_model.pt"}')
        except Exception as e:
            logger.warning(f'Could not save model directly: {e}')

    _write_run_manifest(config, run_name, args, metrics=metrics, training_dir=training_dir)

    logger.info(f'✓ All outputs saved to: {config.OUTPUTS_PATH}')
    logger.info('✓ Generated files:')
    logger.info('  • best_model.pt')
    logger.info('  • metrics.json')
    logger.info('  • classification_metrics.json')
    logger.info('  • training_report.txt')
    logger.info('  • training_metrics.png')
    logger.info('  • run_manifest.json')
    if (config.OUTPUTS_PATH / 'class_distribution.png').exists():
        logger.info('  • class_distribution.png')


if __name__ == '__main__':
    main()


