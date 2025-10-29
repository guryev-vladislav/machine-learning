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
    from datasets.loader import load_data_for_classification
    from models.classifier import build_classifier
    from utils.metrics import save_run_parameters
    from utils.visualization import plot_training_history, plot_confusion_matrix
    from utils.config import (setup_experiment, H5_FILE_PATH, BATCH_SIZE, NUM_CLASSES,
                             EPOCHS, EARLY_STOPPING_MONITOR, EARLY_STOPPING_PATIENCE,
                             EARLY_STOPPING_RESTORE_BEST_WEIGHTS)
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def _prepare_data(x_train, x_val, x_test):
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    input_shape = x_train.shape[1:]
    return x_train, x_val, x_test, input_shape


def _compile_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _train_model(model, x_train, y_train, x_val, y_val):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS
    )

    logger.info(f"Training on {len(x_train)} samples for {EPOCHS} epochs")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return history


def _evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    return test_loss, test_accuracy, predicted_labels, true_labels


def train_classifier_main(experiment_config=None):
    if experiment_config is None:
        experiment_config = setup_experiment()

    model_save_path = experiment_config['classifier']['model_path']
    results_dir = experiment_config['classifier']['results_dir']
    experiment_name = experiment_config['experiment_name']

    logger.info(f"Classifier configuration - Experiment: {experiment_name}, Model: {model_save_path}")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data_for_classification(H5_FILE_PATH)
    x_train, x_val, x_test, input_shape = _prepare_data(x_train, x_val, x_test)

    logger.info(f"Data prepared - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    model = build_classifier(
        input_shape=input_shape,
        num_classes=NUM_CLASSES,
        use_regularization=False
    )
    model = _compile_model(model)
    logger.info("Model architecture compiled")

    history = _train_model(model, x_train, y_train, x_val, y_val)
    logger.info("Training completed")

    model.save(model_save_path)
    logger.info(f"Model saved: {model_save_path}")

    test_loss, test_accuracy, predicted_labels, true_labels = _evaluate_model(model, x_test, y_test)
    logger.info(f"Test accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")

    plot_training_history(history, results_dir)

    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        true_labels, predicted_labels, class_names,
        "Confusion Matrix (Test Data)", "confusion_matrix.png", results_dir
    )

    run_parameters = {
        "EXPERIMENT_NAME": experiment_name,
        "MODEL_TYPE": "Classifier",
        "RUN_TIMESTAMP": experiment_name.split('_')[-1],
        "DATASET_FILENAME": os.path.basename(H5_FILE_PATH),
        "H5_FILE_PATH": H5_FILE_PATH,
        "DATASET_VERSION": experiment_config['dataset_info']['version'],
        "TRAIN_SAMPLES": x_train.shape[0],
        "VAL_SAMPLES": x_val.shape[0],
        "TEST_SAMPLES": x_test.shape[0],
        "INPUT_SHAPE": str(input_shape),
        "NUM_CLASSES": NUM_CLASSES,
        "MODEL_TOTAL_PARAMS": model.count_params(),
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "FINAL_TRAINING_ACCURACY": float(history.history['accuracy'][-1]),
        "FINAL_VALIDATION_ACCURACY": float(history.history['val_accuracy'][-1]),
        "TEST_ACCURACY": float(test_accuracy),
        "TEST_LOSS": float(test_loss),
        "MODEL_SAVE_PATH": model_save_path,
        "RESULTS_DIRECTORY": results_dir,
    }

    save_run_parameters(run_parameters, results_dir, 'classifier_parameters.txt')
    logger.info("All tasks completed")
    return True


def main():
    return train_classifier_main()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train_classifier_main()