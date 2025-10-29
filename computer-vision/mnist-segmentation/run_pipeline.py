# run_pipeline.py
import os
import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def run_full_pipeline():
    """
    Complete training pipeline for U-Net project
    Modify parameters below to configure your training run
    """

    # ==================== CONFIGURATION ====================
    # Modify these parameters as needed

    # Dataset parameters
    DATASET_SIZE = 100  # Number of images to use from MNIST

    # Training parameters
    EPOCHS = 10  # Number of training epochs
    LEARNING_RATE = 0.001  # Learning rate for optimizer

    # Model parameters
    SEGMENTATION_WEIGHT = 0.6  # Weight for segmentation loss
    CLASSIFICATION_WEIGHT = 0.4  # Weight for classification loss
    # =======================================================

    logger.info("Starting complete training pipeline")
    logger.info(f"Configuration: {DATASET_SIZE} images, {EPOCHS} epochs")

    try:
        from utils.config import setup_experiment, update_dataset_config, update_unet_config

        # Update configuration
        update_dataset_config(DATASET_SIZE=DATASET_SIZE)
        update_unet_config(
            EPOCHS=EPOCHS,
            LEARNING_RATE=LEARNING_RATE,
            SEGMENTATION_LOSS_WEIGHT=SEGMENTATION_WEIGHT,
            CLASSIFICATION_LOSS_WEIGHT=CLASSIFICATION_WEIGHT
        )

        # Setup experiment
        experiment_config = setup_experiment()
        logger.info(f"Experiment: {experiment_config['experiment_name']}")

        # Import modules
        from datasets.generator import generate_dataset
        from training.train_classifier import train_classifier_main
        from training.train_unet import train_unet_main

        # Step 1: Generate dataset
        logger.info("Step 1: Generating dataset...")
        start_time = time.time()
        dataset_info = generate_dataset(experiment_config)
        dataset_time = time.time() - start_time
        logger.info(f"Dataset generated in {dataset_time:.1f}s")

        # Step 2: Train classifier
        logger.info("Step 2: Training classifier...")
        start_time = time.time()
        classifier_success = train_classifier_main(experiment_config)
        classifier_time = time.time() - start_time

        if classifier_success:
            logger.info(f"Classifier training completed in {classifier_time:.1f}s")
        else:
            logger.error("Classifier training failed")
            return False

        # Step 3: Train U-Net
        logger.info("Step 3: Training U-Net...")
        start_time = time.time()
        unet_success = train_unet_main(experiment_config)
        unet_time = time.time() - start_time

        if unet_success:
            logger.info(f"U-Net training completed in {unet_time:.1f}s")
        else:
            logger.error("U-Net training failed")
            return False

        total_time = dataset_time + classifier_time + unet_time
        logger.info(f"Complete pipeline finished in {total_time:.1f}s")
        logger.info(f"Results saved to: {experiment_config['experiment_dir']}")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == '__main__':
    run_full_pipeline()