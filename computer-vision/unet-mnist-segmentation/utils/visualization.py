import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix


def plot_training_history(history, save_dir):
    if 'accuracy' in history.history:
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_ran = range(1, len(accuracy) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_ran, accuracy, 'bo-', label='Training Accuracy')
        plt.plot(epochs_ran, val_accuracy, 'b-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_ran, loss, 'ro-', label='Training Loss')
        plt.plot(epochs_ran, val_loss, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    else:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], 'r-', label='Training Total Loss')
        plt.plot(history.history['val_loss'], 'r--', label='Validation Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(history.history['segmentation_output_loss'], 'g-', label='Training Segmentation Loss')
        plt.plot(history.history['val_segmentation_output_loss'], 'g--', label='Validation Segmentation Loss')
        plt.title('Segmentation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(history.history['classification_output_loss'], 'b-', label='Training Classification Loss')
        plt.plot(history.history['val_classification_output_loss'], 'b--', label='Validation Classification Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, title, filename, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()

    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()


def visualize_predictions(model, images, true_masks, true_labels, num_samples=5, save_dir=None, prefix=""):
    plt.figure(figsize=(15, 3 * num_samples))

    indices = np.random.choice(len(images), num_samples, replace=False)

    for i, idx in enumerate(indices):
        single_image = np.expand_dims(images[idx], axis=0)
        true_mask = true_masks[idx]
        true_label = np.argmax(true_labels[idx]) if len(true_labels[idx].shape) > 0 else true_labels[idx]

        predictions = model.predict(single_image, verbose=0)

        if isinstance(predictions, list) and len(predictions) == 2:
            predicted_mask, predicted_class_probs = predictions
            predicted_mask = predicted_mask[0]
            predicted_label = np.argmax(predicted_class_probs[0])
        else:
            predicted_mask = predictions[0] if isinstance(predictions, list) else predictions
            predicted_label = true_label

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(single_image[0, :, :, 0], cmap='gray')
        plt.title(f'Input (True: {true_label})')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(true_mask[:, :, 0], cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(predicted_mask[:, :, 0], cmap='gray')
        plt.title(f'Pred Mask (Class: {predicted_label})')
        plt.axis('off')

    plt.tight_layout()

    if save_dir:
        filename = f'{prefix}predictions.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()