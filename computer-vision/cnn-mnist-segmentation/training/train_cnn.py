# training/train_cnn.py
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
    from models.cnn_classifier import build_cnn_classifier
    from utils.visualization import plot_training_history, plot_confusion_matrix
    from utils.metrics import save_run_parameters
    from utils.config import setup_experiment
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def _compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, early_stopping_patience):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return history


def train_cnn_main(experiment_config=None):
    if experiment_config is None:
        experiment_config = setup_experiment()  # ← ИСПРАВЛЕНО: убрал from utils.cnn_config

    model_save_path = experiment_config['model']['path']
    results_dir = experiment_config['experiment_dir']

    logger.info("Loading MNIST data")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    logger.info(f"Data shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    model = build_cnn_classifier(
        input_shape=(28, 28, 1),
        num_classes=10,
        use_regularization=True
    )

    model = _compile_model(model, learning_rate=0.001)
    logger.info("CNN model compiled")

    history = _train_model(
        model, x_train, y_train, x_val, y_val,
        batch_size=32, epochs=50, early_stopping_patience=5
    )
    logger.info("Training completed")

    model.save(model_save_path)
    logger.info(f"Model saved: {model_save_path}")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    logger.info(f"Test accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")

    plot_training_history(history, results_dir)

    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        true_labels, predicted_labels, class_names,
        "CNN Confusion Matrix", "cnn_confusion_matrix.png", results_dir
    )

    run_parameters = {
        "EXPERIMENT_NAME": experiment_config['experiment_name'],
        "MODEL_TYPE": "CNN Classifier",
        "TEST_ACCURACY": float(test_accuracy),
        "TEST_LOSS": float(test_loss),
        "MODEL_SAVE_PATH": model_save_path,
    }

    save_run_parameters(run_parameters, results_dir, 'cnn_parameters.txt')
    logger.info("CNN training pipeline completed")
    return True


def main():
    return train_cnn_main()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train_cnn_main()