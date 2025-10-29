import tensorflow as tf


def create_dataset(images, masks=None, labels=None, batch_size=32, shuffle=True):
    if masks is not None and labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((images, masks, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        dataset = dataset.map(
            lambda img, msk, lbl: (img, {'segmentation_output': msk, 'classification_output': lbl}),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    elif labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
    else:
        raise ValueError("Either (images, labels) for classification or (images, masks, labels) for segmentation required")

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset