from tensorflow.keras import layers, models


def build_unet(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)

    # Classification Branch
    global_pool = layers.GlobalAveragePooling2D()(conv4)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification_output')(global_pool)

    # Decoder
    up6 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(conv4)
    merge6 = layers.concatenate([conv3, up6])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv6)
    merge7 = layers.concatenate([conv2, up7])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv7)
    merge8 = layers.concatenate([conv1, up8])
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv8)

    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation_output')(conv8)

    model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])
    return model