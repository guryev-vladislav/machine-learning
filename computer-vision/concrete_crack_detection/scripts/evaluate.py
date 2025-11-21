# scripts/evaluate.py
import logging
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.inference.pipeline import InferencePipeline
    from src.data_loaders.deepcrack_loader import DeepCrackDataset
    from src.data_loaders.metu_loader import METUDataset
    from src.utils.transforms import get_test_transforms
    from src.utils.experiment_tracker import ExperimentTracker
except ImportError as e:
    print(f"Import error: {e}")
    raise


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', type=str, required=True, help='Model version to evaluate')
    parser.add_argument('--dataset', type=str, default='deepcrack', choices=['deepcrack', 'metu'])
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    try:
        model_path = Path("models") / args.model_version
        if not model_path.exists():
            raise ValueError(f"Model {args.model_version} not found")

        with open(model_path / "config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        tracker = ExperimentTracker()
        result_folder = f"result_{args.model_version}_eval"
        result_path = Path("results") / result_folder
        result_path.mkdir(parents=True, exist_ok=True)

        if args.dataset == 'deepcrack':
            dataset = DeepCrackDataset(
                data_dir='data/external/deepcrack',
                transform=get_test_transforms(),
                is_train=False
            )
        else:
            dataset = METUDataset(
                data_dir='data/external/metu',
                transform=get_test_transforms(),
                is_train=False
            )

        pipeline = InferencePipeline(model_path / "model.pth")
        metrics, predictions = pipeline.evaluate_dataset(dataset, result_path)
        pipeline.generate_report(metrics, result_path)

        logger.info(f"Evaluation completed for {args.model_version}")
        logger.info(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()