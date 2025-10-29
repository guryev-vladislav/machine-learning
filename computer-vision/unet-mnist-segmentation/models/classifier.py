from tensorflow.keras import Sequential, Input, layers
import tensorflow as tf


def build_classifier(input_shape, num_classes=10, use_regularization=True,
                     l1_reg=0.0001, l2_reg=0.0001, dropout_rate=0.2):
    """Create a simple feedforward classifier with regularization options"""
    regularizer = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg) if use_regularization else None

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