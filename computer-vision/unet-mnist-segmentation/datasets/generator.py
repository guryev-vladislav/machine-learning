# datasets/generator.py
import tensorflow as tf
import numpy as np
import h5py
import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.config import H5_FILE_PATH, get_dataset_config
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def _validate_dataset_parameters(dataset_config):
    """Validate dataset generation parameters"""
    ratios_sum = dataset_config['TRAIN_RATIO'] + dataset_config['VAL_RATIO'] + dataset_config['TEST_RATIO']
    assert abs(ratios_sum - 1.0) < 1e-10, f"Ratio sum must be 1.0, got {ratios_sum}"
    assert dataset_config['DATASET_SIZE'] > 0, "Dataset size must be positive"
    assert all(s > 0 for s in dataset_config['OUTPUT_IMAGE_SIZE']), "Image size must be positive"
    assert len(dataset_config['AUGMENTATION_FACTORS']) > 0, "At least one scale factor must be provided"


def _augment_image_and_create_mask(image, label, scale_factor, output_image_size, num_classes):
    """Helper function for image augmentation and mask creation"""
    original_h, original_w = tf.shape(image)[0], tf.shape(image)[1]
    new_h = tf.cast(tf.cast(original_h, tf.float32) * scale_factor, tf.int32)
    new_w = tf.cast(tf.cast(original_w, tf.float32) * scale_factor, tf.int32)

    scaled_image = tf.image.resize(image, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)

    start_h = tf.cast((output_image_size[0] - new_h) / 2, tf.int32)
    start_w = tf.cast((output_image_size[1] - new_w) / 2, tf.int32)

    pad_top = tf.maximum(0, start_h)
    pad_bottom = tf.maximum(0, output_image_size[0] - (start_h + new_h))
    pad_left = tf.maximum(0, start_w)
    pad_right = tf.maximum(0, output_image_size[1] - (start_w + new_w))

    padded_scaled_image = tf.pad(scaled_image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    crop_start_h = tf.maximum(0, -start_h)
    crop_start_w = tf.maximum(0, -start_w)

    final_image = tf.image.crop_to_bounding_box(
        padded_scaled_image,
        offset_height=crop_start_h,
        offset_width=crop_start_w,
        target_height=output_image_size[0],
        target_width=output_image_size[1]
    )

    mask = tf.where(final_image > 0.1, 1.0, 0.0)
    one_hot_label = tf.one_hot(label, depth=num_classes)

    return final_image, mask, one_hot_label


def _process_dataset_split(x_split, y_split, split_name, dataset_config, num_classes):
    """Process one dataset split"""
    x_processed, masks_processed, y_processed = [], [], []

    logger.info(f"Processing {split_name} split ({len(x_split)} original images)")

    for i, (original_img, original_lbl) in enumerate(zip(x_split, y_split)):
        if i % 100 == 0:
            logger.debug(f"{split_name}: processed {i}/{len(x_split)} images")

        for scale_factor in dataset_config['AUGMENTATION_FACTORS']:
            img_tensor = tf.constant(original_img, dtype=tf.float32)
            lbl_tensor = tf.constant(original_lbl, dtype=tf.int32)
            augmented_img, mask, one_hot_lbl = _augment_image_and_create_mask(
                img_tensor, lbl_tensor, scale_factor,
                dataset_config['OUTPUT_IMAGE_SIZE'], num_classes
            )

            x_processed.append(augmented_img.numpy())
            masks_processed.append(mask.numpy())
            y_processed.append(one_hot_lbl.numpy())

    return np.array(x_processed, dtype=np.float32), np.array(masks_processed, dtype=np.float32), np.array(y_processed,
                                                                                                          dtype=np.float32)


def generate_dataset(experiment_config=None):
    """Main function for generating and saving augmented dataset"""

    # Get dataset config
    if experiment_config and 'configs' in experiment_config:
        dataset_config = experiment_config['configs']['dataset']
    else:
        dataset_config = get_dataset_config()

    _validate_dataset_parameters(dataset_config)

    logger.info("Loading original MNIST dataset...")
    (x_train_original, y_train_original), (x_test_original, y_test_original) = tf.keras.datasets.mnist.load_data()
    logger.info("Original MNIST dataset loaded successfully")

    # Combine and shuffle data
    x_combined = np.concatenate((x_train_original, x_test_original), axis=0)
    y_combined = np.concatenate((y_train_original, y_test_original), axis=0)

    # Process images
    x_combined = np.expand_dims(x_combined, axis=-1)  # Add channel dimension
    if dataset_config['NORMALIZE']:
        x_combined = x_combined.astype('float32') / 255.0
    else:
        x_combined = x_combined.astype('float32')

    # Limit size and shuffle
    actual_total_samples = min(dataset_config['DATASET_SIZE'], len(x_combined))
    indices = np.random.permutation(len(x_combined))[:actual_total_samples]
    x_selected = x_combined[indices]
    y_selected = y_combined[indices]

    # Split data
    num_train = int(actual_total_samples * dataset_config['TRAIN_RATIO'])
    num_val = int(actual_total_samples * dataset_config['VAL_RATIO'])

    x_train = x_selected[:num_train]
    y_train = y_selected[:num_train]
    x_val = x_selected[num_train:num_train + num_val]
    y_val = y_selected[num_train:num_train + num_val]
    x_test = x_selected[num_train + num_val:]
    y_test = y_selected[num_train + num_val:]

    logger.info(f"Generating dataset from {actual_total_samples} original images")
    logger.info(
        f"Train: {len(x_train)} * {len(dataset_config['AUGMENTATION_FACTORS'])} = {len(x_train) * len(dataset_config['AUGMENTATION_FACTORS'])}")
    logger.info(
        f"Val: {len(x_val)} * {len(dataset_config['AUGMENTATION_FACTORS'])} = {len(x_val) * len(dataset_config['AUGMENTATION_FACTORS'])}")
    logger.info(
        f"Test: {len(x_test)} * {len(dataset_config['AUGMENTATION_FACTORS'])} = {len(x_test) * len(dataset_config['AUGMENTATION_FACTORS'])}")

    # Process all splits
    num_classes = 10  # MNIST has 10 classes
    x_train_final, mask_train_final, y_train_final = _process_dataset_split(x_train, y_train, "train", dataset_config,
                                                                            num_classes)
    x_val_final, mask_val_final, y_val_final = _process_dataset_split(x_val, y_val, "validation", dataset_config,
                                                                      num_classes)
    x_test_final, mask_test_final, y_test_final = _process_dataset_split(x_test, y_test, "test", dataset_config,
                                                                         num_classes)

    # Save dataset
    os.makedirs(os.path.dirname(H5_FILE_PATH), exist_ok=True)

    logger.info(f"Saving dataset to: {H5_FILE_PATH}")
    with h5py.File(H5_FILE_PATH, 'w') as f:
        f.create_dataset('x_train', data=x_train_final, compression="gzip", compression_opts=9)
        f.create_dataset('y_train', data=y_train_final, compression="gzip", compression_opts=9)
        f.create_dataset('mask_train', data=mask_train_final, compression="gzip", compression_opts=9)
        f.create_dataset('x_val', data=x_val_final, compression="gzip", compression_opts=9)
        f.create_dataset('y_val', data=y_val_final, compression="gzip", compression_opts=9)
        f.create_dataset('mask_val', data=mask_val_final, compression="gzip", compression_opts=9)
        f.create_dataset('x_test', data=x_test_final, compression="gzip", compression_opts=9)
        f.create_dataset('y_test', data=y_test_final, compression="gzip", compression_opts=9)
        f.create_dataset('mask_test', data=mask_test_final, compression="gzip", compression_opts=9)

    dataset_info = {
        'total_original_samples': actual_total_samples,
        'total_augmented_samples': len(x_train_final) + len(x_val_final) + len(x_test_final),
        'output_path': H5_FILE_PATH,
        'config_used': dataset_config,
        'shapes': {
            'x_train': x_train_final.shape,
            'mask_train': mask_train_final.shape,
            'y_train': y_train_final.shape,
            'x_val': x_val_final.shape,
            'mask_val': mask_val_final.shape,
            'y_val': y_val_final.shape,
            'x_test': x_test_final.shape,
            'mask_test': mask_test_final.shape,
            'y_test': y_test_final.shape
        }
    }

    logger.info("Dataset successfully generated")
    return dataset_info


if __name__ == '__main__':
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    dataset_info = generate_dataset()
    logger.info(f"Dataset info: {dataset_info}")