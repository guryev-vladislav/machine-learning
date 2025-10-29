# models/cnn_classifier.py
from tensorflow.keras import Sequential, Input, layers
import tensorflow as tf


def build_cnn_classifier(input_shape=(28, 28, 1), num_classes=10, use_regularization=True,
                        l1_reg=0.0001, l2_reg=0.0001, dropout_rate=0.2):
    regularizer = None
    if use_regularization:
        regularizer = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)

    model = Sequential([
        Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model