import tensorflow as tf
import numpy as np
import os
import sys
import logging

logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from datasets.loader import load_data_for_unet
    from models.unet import build_unet
    from utils.visualization import plot_training_history, visualize_predictions, plot_confusion_matrix
    from utils.metrics import save_run_parameters
    from utils.config import (setup_experiment, H5_FILE_PATH, get_unet_config)
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def create_dataset(images, masks, labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))

    dataset = dataset.map(
        lambda img, msk, lbl: (img, {'segmentation_output': msk, 'classification_output': lbl}),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def _compile_model(model, unet_config):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=unet_config['LEARNING_RATE']),
        loss={
            'segmentation_output': 'binary_crossentropy',
            'classification_output': 'categorical_crossentropy'
        },
        metrics={
            'segmentation_output': ['accuracy'],
            'classification_output': ['accuracy']
        },
        loss_weights={
            'segmentation_output': unet_config['SEGMENTATION_LOSS_WEIGHT'],
            'classification_output': unet_config['CLASSIFICATION_LOSS_WEIGHT']
        }
    )
    return model


def _create_callbacks(unet_config):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=unet_config['EARLY_STOPPING_MONITOR'],
            patience=unet_config['EARLY_STOPPING_PATIENCE'],
            restore_best_weights=unet_config['EARLY_STOPPING_RESTORE_BEST_WEIGHTS'],
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=unet_config['REDUCE_LR_MONITOR'],
            factor=unet_config['REDUCE_LR_FACTOR'],
            patience=unet_config['REDUCE_LR_PATIENCE'],
            verbose=1,
            min_lr=unet_config['REDUCE_LR_MIN_LR']
        )
    ]


def _train_model(model, train_ds, val_ds, unet_config):
    callbacks = _create_callbacks(unet_config)

    logger.info(f"Starting U-Net training - Epochs: {unet_config['EPOCHS']}, Batch size: {unet_config['BATCH_SIZE']}")

    history = model.fit(
        train_ds,
        epochs=unet_config['EPOCHS'],
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    return history


def _evaluate_model(model, test_ds):
    evaluation = model.evaluate(test_ds, verbose=0, return_dict=True)

    logger.info("Test results:")
    for metric, value in evaluation.items():
        logger.info(f"{metric}: {value:.4f}")

    return evaluation


def _get_predictions(model, x_test):
    predictions = model.predict(x_test, verbose=0)

    if isinstance(predictions, list) and len(predictions) == 2:
        mask_predictions, class_predictions = predictions
        predicted_labels = np.argmax(class_predictions, axis=1)
        return mask_predictions, class_predictions, predicted_labels
    else:
        raise ValueError("Unexpected model predictions format")


def train_unet_main(experiment_config=None):
    if experiment_config is None:
        experiment_config = setup_experiment()

    if experiment_config and 'configs' in experiment_config:
        unet_config = experiment_config['configs']['unet']
    else:
        unet_config = get_unet_config()

    model_save_path = experiment_config['model']['path']
    experiment_dir = experiment_config['experiment_dir']
    experiment_name = experiment_config['experiment_name']

    logger.info(f"U-Net configuration - Experiment: {experiment_name}, Model: {model_save_path}")

    try:
        (x_train, mask_train, y_train), (x_val, mask_val, y_val), (x_test, mask_test, y_test) = load_data_for_unet(H5_FILE_PATH)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return False

    logger.info(f"Data shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    try:
        train_ds = create_dataset(x_train, mask_train, y_train, unet_config['BATCH_SIZE'], shuffle=True)
        val_ds = create_dataset(x_val, mask_val, y_val, unet_config['BATCH_SIZE'], shuffle=False)
        test_ds = create_dataset(x_test, mask_test, y_test, unet_config['BATCH_SIZE'], shuffle=False)
        logger.info("Datasets created successfully")
    except Exception as e:
        logger.error(f"Dataset creation error: {e}")
        return False

    try:
        model = build_unet(unet_config['INPUT_SHAPE'], unet_config['NUM_CLASSES'])
        model = _compile_model(model, unet_config)
        logger.info("Model built and compiled")
    except Exception as e:
        logger.error(f"Model building error: {e}")
        return False

    try:
        history = _train_model(model, train_ds, val_ds, unet_config)
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False

    try:
        model.save(model_save_path)
        logger.info(f"Model saved: {model_save_path}")
    except Exception as e:
        logger.error(f"Model saving error: {e}")
        return False

    try:
        test_results = _evaluate_model(model, test_ds)
        _, _, predicted_labels = _get_predictions(model, x_test)
        true_labels = np.argmax(y_test, axis=1)
        logger.info("Evaluation completed")
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return False

    try:
        class_names = [str(i) for i in range(unet_config['NUM_CLASSES'])]

        plot_confusion_matrix(
            true_labels, predicted_labels, class_names,
            'Confusion Matrix (U-Net Classification)', 'confusion_matrix.png', experiment_dir
        )

        visualize_predictions(
            model, x_test, mask_test, y_test,
            num_samples=5, save_dir=experiment_dir, prefix="unet_"
        )

        plot_training_history(history, experiment_dir)
        logger.info("Visualization completed")
    except Exception as e:
        logger.warning(f"Visualization error: {e}")

    try:
        run_parameters = {
            "EXPERIMENT_NAME": experiment_name,
            "MODEL_TYPE": "U-Net Multi-Output",
            "RUN_TIMESTAMP": experiment_name.split('_')[-1],
            "DATASET_FILENAME": os.path.basename(H5_FILE_PATH),
            "H5_FILE_PATH": H5_FILE_PATH,
            "DATASET_VERSION": experiment_config['dataset_info']['version'],
            "TRAIN_SAMPLES": x_train.shape[0],
            "VAL_SAMPLES": x_val.shape[0],
            "TEST_SAMPLES": x_test.shape[0],
            "INPUT_SHAPE": str(unet_config['INPUT_SHAPE']),
            "NUM_CLASSES": unet_config['NUM_CLASSES'],
            "MODEL_TOTAL_PARAMS": model.count_params(),
            "LEARNING_RATE": unet_config['LEARNING_RATE'],
            "BATCH_SIZE": unet_config['BATCH_SIZE'],
            "EPOCHS": unet_config['EPOCHS'],
            "SEGMENTATION_LOSS_WEIGHT": unet_config['SEGMENTATION_LOSS_WEIGHT'],
            "CLASSIFICATION_LOSS_WEIGHT": unet_config['CLASSIFICATION_LOSS_WEIGHT'],
            "TOTAL_TRAINING_LOSS_FINAL": float(history.history['loss'][-1]),
            "TOTAL_VALIDATION_LOSS_FINAL": float(history.history['val_loss'][-1]),
            "TEST_SEGMENTATION_ACCURACY": float(test_results.get('segmentation_output_accuracy', 0)),
            "TEST_CLASSIFICATION_ACCURACY": float(test_results.get('classification_output_accuracy', 0)),
            "TEST_SEGMENTATION_LOSS": float(test_results.get('segmentation_output_loss', 0)),
            "TEST_CLASSIFICATION_LOSS": float(test_results.get('classification_output_loss', 0)),
            "TEST_TOTAL_LOSS": float(test_results.get('loss', 0)),
            "MODEL_SAVE_PATH": model_save_path,
            "RESULTS_DIRECTORY": experiment_dir,
        }

        save_run_parameters(run_parameters, experiment_dir, 'unet_parameters.txt')
        logger.info("Parameters saved")
    except Exception as e:
        logger.error(f"Parameter saving error: {e}")
        return False

    logger.info("U-Net training completed successfully")
    return True


def main():
    return train_unet_main()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train_unet_main()