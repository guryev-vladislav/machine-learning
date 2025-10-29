# run_cnn_pipeline.py
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def run_cnn_training():
    # ==================== CONFIGURATION ====================
    epochs = 50
    batch_size = 32
    # =======================================================

    logger.info(f"Starting CNN training - Epochs: {epochs}, Batch size: {batch_size}")

    try:
        from training.train_cnn import train_cnn_main

        start_time = time.time()
        success = train_cnn_main()
        training_time = time.time() - start_time

        if success:
            logger.info(f"CNN training completed in {training_time:.1f}s")
        else:
            logger.error("CNN training failed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == '__main__':
    run_cnn_training()