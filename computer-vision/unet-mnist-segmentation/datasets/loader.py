import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_data_for_unet(h5_file_path):
    """Load data for U-Net (with masks)"""
    logger.info(f"Loading U-Net data from HDF5 file: {h5_file_path}")

    with h5py.File(h5_file_path, 'r') as f:
        x_train = np.array(f['x_train'])
        mask_train = np.array(f['mask_train'])
        y_train = np.array(f['y_train'])
        x_val = np.array(f['x_val'])
        mask_val = np.array(f['mask_val'])
        y_val = np.array(f['y_val'])
        x_test = np.array(f['x_test'])
        mask_test = np.array(f['mask_test'])
        y_test = np.array(f['y_test'])

    logger.info("Data successfully loaded")
    logger.info(f"Dataset shapes - "
                f"Train: x{x_train.shape}, mask{mask_train.shape}, y{y_train.shape} | "
                f"Val: x{x_val.shape}, mask{mask_val.shape}, y{y_val.shape} | "
                f"Test: x{x_test.shape}, mask{mask_test.shape}, y{y_test.shape}")

    return (x_train, mask_train, y_train), (x_val, mask_val, y_val), (x_test, mask_test, y_test)


def load_data_for_classification(h5_file_path):
    """Load data for classification only (without masks)"""
    logger.info(f"Loading classification data from: {h5_file_path}")

    with h5py.File(h5_file_path, 'r') as f:
        x_train = np.array(f['x_train'])
        y_train = np.array(f['y_train'])
        x_val = np.array(f['x_val'])
        y_val = np.array(f['y_val'])
        x_test = np.array(f['x_test'])
        y_test = np.array(f['y_test'])

    logger.info("Classification data successfully loaded")
    logger.info(f"Classification dataset shapes - "
                f"Train: x{x_train.shape}, y{y_train.shape} | "
                f"Val: x{x_val.shape}, y{y_val.shape} | "
                f"Test: x{x_test.shape}, y{y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)