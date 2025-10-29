# utils/config.py
import os
import datetime
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'

MODELS_SAVED_DIR = os.path.join(BASE_DIR, MODELS_FOLDER, 'saved')
RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_FOLDER)

CNN_CONFIG = {
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'EARLY_STOPPING_PATIENCE': 5,
}

def create_directories():
    directories = [MODELS_SAVED_DIR, RESULTS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

def get_experiment_name():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"cnn_experiment_{timestamp}"

def setup_experiment():
    EXPERIMENT_NAME = get_experiment_name()
    EXPERIMENT_DIR = os.path.join(RESULTS_DIR, EXPERIMENT_NAME)

    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        logger.info(f"Created experiment directory: {EXPERIMENT_DIR}")

    model_path = os.path.join(MODELS_SAVED_DIR, f"cnn_{EXPERIMENT_NAME}.keras")

    experiment_config = {
        'experiment_name': EXPERIMENT_NAME,
        'experiment_dir': EXPERIMENT_DIR,
        'model': {
            'path': model_path,
            'type': 'CNN Classifier'
        },
        'config': CNN_CONFIG
    }

    return experiment_config

create_directories()