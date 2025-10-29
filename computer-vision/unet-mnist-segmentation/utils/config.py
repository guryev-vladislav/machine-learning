# utils/config.py
import os
import glob
import datetime
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FOLDER = 'datasets'
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'

DATASET_GENERATED_DIR = os.path.join(BASE_DIR, DATASET_FOLDER, 'generated')
MODELS_SAVED_DIR = os.path.join(BASE_DIR, MODELS_FOLDER, 'saved')
RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_FOLDER)

# Общие константы для всех моделей
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 50
EARLY_STOPPING_MONITOR = 'val_accuracy'
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True

DATASET_CONSTANTS = {
    'TRAIN_RATIO': 0.7,
    'VAL_RATIO': 0.15,
    'TEST_RATIO': 0.15,
    'AUGMENTATION_FACTORS': (0.5, 1.0, 2.0, 4.0),
    'OUTPUT_IMAGE_SIZE': (112, 112),
    'NORMALIZE': True,
    'GRAYSCALE': True,
}

UNET_CONSTANTS = {
    'INPUT_SHAPE': (112, 112, 1),
    'NUM_CLASSES': NUM_CLASSES,  # Используем общую константу
    'EARLY_STOPPING_MONITOR': 'val_loss',
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'EARLY_STOPPING_RESTORE_BEST_WEIGHTS': EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    'REDUCE_LR_MONITOR': 'val_loss',
    'REDUCE_LR_FACTOR': 0.5,
    'REDUCE_LR_PATIENCE': 5,
    'REDUCE_LR_MIN_LR': 1e-7,
}

DATASET_CONFIG = {
    'DATASET_SIZE': 1000,
}

UNET_CONFIG = {
    'BATCH_SIZE': BATCH_SIZE,  # Используем общую константу
    'EPOCHS': EPOCHS,          # Используем общую константу
    'LEARNING_RATE': 0.001,
    'SEGMENTATION_LOSS_WEIGHT': 0.6,
    'CLASSIFICATION_LOSS_WEIGHT': 0.4,
}


def get_dataset_config():
    return {**DATASET_CONSTANTS, **DATASET_CONFIG}


def get_unet_config():
    return {**UNET_CONSTANTS, **UNET_CONFIG}


def update_dataset_config(**updates):
    global DATASET_CONFIG
    DATASET_CONFIG.update(updates)


def update_unet_config(**updates):
    global UNET_CONFIG
    UNET_CONFIG.update(updates)


def create_directories():
    directories = [
        DATASET_GENERATED_DIR,
        MODELS_SAVED_DIR,
        RESULTS_DIR
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


def get_next_dataset_version():
    if not os.path.exists(DATASET_GENERATED_DIR):
        os.makedirs(DATASET_GENERATED_DIR, exist_ok=True)

    pattern = os.path.join(DATASET_GENERATED_DIR, 'synthetic_mnist_v*.h5')
    existing_files = glob.glob(pattern)

    if not existing_files:
        return 1

    versions = []
    for file in existing_files:
        try:
            version = int(file.split('_v')[-1].split('.h5')[0])
            versions.append(version)
        except:
            continue

    return max(versions) + 1 if versions else 1


def get_experiment_name():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_version = get_next_dataset_version()
    return f"unet_experiment_v{dataset_version}_{timestamp}"


DATASET_VERSION = get_next_dataset_version()
DATASET_FILENAME = f'synthetic_mnist_v{DATASET_VERSION}.h5'
H5_FILE_PATH = os.path.join(DATASET_GENERATED_DIR, DATASET_FILENAME)


def setup_experiment():
    EXPERIMENT_NAME = get_experiment_name()
    EXPERIMENT_DIR = os.path.join(RESULTS_DIR, EXPERIMENT_NAME)

    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        logger.info(f"Created experiment directory: {EXPERIMENT_DIR}")

    unet_model_path = os.path.join(MODELS_SAVED_DIR, f"unet_{EXPERIMENT_NAME}.keras")
    classifier_model_path = os.path.join(MODELS_SAVED_DIR, f"classifier_{EXPERIMENT_NAME}.keras")

    experiment_config = {
        'experiment_name': EXPERIMENT_NAME,
        'experiment_dir': EXPERIMENT_DIR,
        'model': {
            'path': unet_model_path,
            'type': 'U-Net Multi-Output'
        },
        'classifier': {
            'model_path': classifier_model_path,
            'results_dir': EXPERIMENT_DIR
        },
        'dataset_info': {
            'filename': DATASET_FILENAME,
            'path': H5_FILE_PATH,
            'version': DATASET_VERSION
        },
        'configs': {
            'dataset': get_dataset_config(),
            'unet': get_unet_config()
        }
    }

    logger.info(f"Experiment setup complete: {EXPERIMENT_NAME}")
    return experiment_config


create_directories()