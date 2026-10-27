"""Evaluation pipeline: тестирование нейросети на разных материалах.

Нейросеть (YOLO) — только детектор трещин: crack vs no_crack.
Материал определяет алгоритмический слой (отдельно).

Usage:
    python evaluate.py --model outputs/yolo_crack_detection/weights/best.pt

    Показывает Recall по каждому материалу:
    - Concrete: насколько хорошо находит трещины на бетоне
    - Wood: насколько хорошо находит трещины на дереве
    - No crack: насколько хорошо говорит "нет трещин"
"""

import argparse
import sys
from pathlib import Path

# Add project root to PYTHONPATH
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logging_setup import get_logger, shutdown_logging

logger = get_logger(__name__)


def evaluate_model(model_path, dataset_root):
    """Run evaluation on test categories.

    Metrics per material:
    - Recall (crack): TP / (TP + FN) — насколько хорошо находит трещины
    - Precision (no_crack): TN / (TN + FP) — насколько хорошо пропускает без трещин
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error('ultralytics is required for evaluation')
        return

    logger.info(f'Loading model: {model_path}')
    model = YOLO(str(model_path))

    test_dir = Path(dataset_root) / 'test'
    if not test_dir.exists():
        logger.error(f'Test directory not found: {test_dir}')
        logger.info('Create data/dataset/test/{category}/ for each material')
        return

    categories = [d.name for d in test_dir.iterdir() if d.is_dir()]
    if not categories:
        logger.error(f'No categories found in {test_dir}')
        return

    logger.info(f'Categories found: {categories}')
    logger.info(
        f'\n{"="*70}\n'
        f'NEURAL NETWORK EVALUATION\n'
        f'(Crack Detection Only — Material Classification is Separate)\n'
        f'{"="*70}\n'
    )

    # Results storage
    all_results = {}

    for category in sorted(categories):
        cat_path = test_dir / category
        images = list(cat_path.glob('*'))
        images = [f for f in images if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

        if not images:
            logger.warning(f'No images in {cat_path}, skipping')
            continue

        # Ground truth label
        is_crack_category = 'crack' in category.lower() and 'no' not in category.lower()
        gt_label = 'crack' if is_crack_category else 'no_crack'

        logger.info(f'\n{"─"*70}')
        logger.info(f'Material: {category}')
        logger.info(f'Ground truth: {"CRACK" if is_crack_category else "NO CRACK"}')
        logger.info(f'Images: {len(images)}')
        logger.info(f'{"─"*70}')

        # Run inference
        predictions = model.predict(
            source=str(images),
            conf=0.5,
            verbose=False,
            imgsz=416,
        )

        # Calculate metrics
        tp = 0  # True Positive: crack exists, predicted crack
        fp = 0  # False Positive: no crack, predicted crack
        tn = 0  # True Negative: no crack, predicted no_crack
        fn = 0  # False Negative: crack exists, predicted no_crack

        for i, pred in enumerate(predictions):
            probs = pred.probs
            if probs is None:
                continue

            pred_label = 'crack' if probs.top1 == 1 else 'no_crack'

            if is_crack_category:
                if pred_label == 'crack':
                    tp += 1
                    status = '✅'
                else:
                    fn += 1
                    status = '❌ MISSED'
            else:
                if pred_label == 'no_crack':
                    tn += 1
                    status = '✅'
                else:
                    fp += 1
                    status = '❌ FALSE'

            if i < 3:  # Show first 3 examples
                img_name = Path(pred.orig_path).stem
                logger.info(
                    f'  {status} {img_name}: pred={pred_label}, '
                    f'gt={gt_label}'
                )

        # Calculate metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        all_results[category] = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'recall': recall, 'precision': precision,
            'specificity': specificity, 'accuracy': accuracy,
            'total': tp + fp + tn + fn,
        }

        logger.info(f'\n  METRICS:')
        logger.info(f'    Recall (crack detection):    {recall:.2%}  '
                    f'({tp}/{tp + fn} трещин найдено)')
        logger.info(f'    Precision (positive):        {precision:.2%}  '
                    f'({tp} из {tp + fp} предсказаний верно)')
        logger.info(f'    Specificity (no_crack):      {specificity:.2%}  '
                    f'({tn}/{tn + fp} без трещин верно)')
        logger.info(f'    Accuracy:                    {accuracy:.2%}')

        if not is_crack_category:
            logger.info(f'    (Для no_crack важнее Specificity — не ложные срабатывания)')
        else:
            logger.info(f'    (Для crack важнее Recall — не пропускать трещины)')

    # Summary table
    logger.info(f'\n{"="*70}')
    logger.info('SUMMARY — NEURAL NETWORK (CRACK DETECTION ONLY)')
    logger.info(f'{"="*70}')
    logger.info(
        f'{"Material":<20} {"Recall":<10} {"Precision":<10} '
        f'{"Specificity":<12} {"Accuracy"}'
    )
    logger.info(f'{"─"*70}')

    for cat, m in sorted(all_results.items()):
        logger.info(
            f'{cat:<20} {m["recall"]:<10.2%} {m["precision"]:<10.2%} '
            f'{m["specificity"]:<12.2%} {m["accuracy"]:.2%}'
        )

    # Overall
    total_tp = sum(m['tp'] for m in all_results.values())
    total_fn = sum(m['fn'] for m in all_results.values())
    total_tn = sum(m['tn'] for m in all_results.values())
    total_fp = sum(m['fp'] for m in all_results.values())

    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)

    logger.info(f'{"─"*70}')
    logger.info(
        f'{"OVERALL":<20} {overall_recall:<10.2%} '
        f'{(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0:<10.2%} '
        f'{(total_tn / (total_tn + total_fp)) if (total_tn + total_fp) > 0 else 0:<12.2%} '
        f'{overall_accuracy:.2%}'
    )

    # Interpretation
    logger.info(f'\n{"="*70}')
    logger.info('INTERPRETATION')
    logger.info(f'{"="*70}')
    logger.info(
        f'Neural network detects cracks (crack vs no_crack).\n'
        f'Material classification (concrete/wood/metal) is handled by\n'
        f'a separate algorithmic layer.\n\n'
        f'Key metrics:\n'
        f'  • Recall > 90% = neural network finds most cracks\n'
        f'  • Low Recall on wood = model needs more wood training data\n'
        f'  • Specificity > 95% = few false positives on clean surfaces'
    )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO crack detection on multiple materials'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (best.pt)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=str(_project_root / 'data' / 'dataset'),
        help='Path to dataset directory (default: data/dataset)',
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f'Model not found: {model_path}')
        return

    evaluate_model(args.model, args.dataset)
    shutdown_logging()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info('Evaluation cancelled by user')
        shutdown_logging()
