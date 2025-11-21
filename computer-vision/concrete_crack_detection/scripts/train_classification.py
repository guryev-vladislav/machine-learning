# scripts/train_classification.py
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
    from src.data_loaders.sdnet_loader import SDNETDataset
    from src.utils.transforms import get_train_transforms, get_val_transforms
    from src.utils.experiment_tracker import ExperimentTracker
    from src.training.classification_trainer import ClassificationTrainer
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

        train_dataset = SDNETDataset(
            data_dir='data/external/sdnet2018',
            transform=get_train_transforms(),
            is_train=True
        )

        val_dataset = SDNETDataset(
            data_dir='data/external/sdnet2018',
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
            model_type="classifier",
            num_classes=2
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate']
        )

        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )

        logger.info("Starting classification training")

        for epoch in range(config['training']['epochs']):
            train_loss, train_acc, train_f1 = trainer.train_epoch()
            val_loss, val_acc, val_f1 = trainer.validate_epoch()

            logger.info(f'Epoch {epoch + 1}')
            logger.info(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')

            if val_acc > trainer.best_metric:
                trainer.best_metric = val_acc
                trainer.save_checkpoint(model_path, epoch, val_acc)

        logger.info(f"Classification training completed. Model: {model_version}")

    except Exception as e:
        logger.error(f"Classification training failed: {e}")
        raise


if __name__ == "__main__":
    main()