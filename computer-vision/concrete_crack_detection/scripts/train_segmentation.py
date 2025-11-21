# scripts/train_segmentation.py
import torch
import torch.nn as nn
import logging
import yaml
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.model_factory import ModelFactory
    from src.data_loaders.deepcrack_loader import DeepCrackDataset
    from src.utils.transforms import get_train_transforms, get_val_transforms
    from src.utils.experiment_tracker import ExperimentTracker
    from src.training.segmentation_trainer import SegmentationTrainer
except ImportError as e:
    print(f"Import error: {e}")
    raise


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        with open('configs/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        tracker = ExperimentTracker()
        model_version = tracker.get_next_model_version()
        model_path, result_path = tracker.create_experiment_folders(model_version)
        tracker.save_metadata(result_path, config)

        train_dataset = DeepCrackDataset(
            data_dir='data/external/deepcrack',
            transform=get_train_transforms(),
            is_train=True
        )

        val_dataset = DeepCrackDataset(
            data_dir='data/external/deepcrack',
            transform=get_val_transforms(),
            is_train=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['training']['batch_size'], shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['training']['batch_size'], shuffle=False
        )

        model = ModelFactory.create_model(
            model_type=config['model']['architecture'],
            in_channels=3,
            out_channels=config['model']['num_classes']
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )

        trainer.train(
            epochs=config['training']['epochs'],
            model_path=model_path,
            result_path=result_path
        )

        logger.info(f"Training completed. Model saved as {model_version}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()