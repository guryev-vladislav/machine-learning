import os


def save_run_parameters(params, save_dir, filename='parameters.txt'):
    filepath = os.path.join(save_dir, filename)

    def safe_float_format(value, default='N/A'):
        if value == 'N/A' or value is None:
            return default
        try:
            return f"{float(value):.4f}"
        except (ValueError, TypeError):
            return str(value)

    with open(filepath, 'w') as f:
        f.write("--- Run Parameters ---\n")
        f.write(f"Experiment Name: {params.get('EXPERIMENT_NAME', 'N/A')}\n")
        f.write(f"Model Type: {params.get('MODEL_TYPE', 'N/A')}\n")
        f.write(f"Run Timestamp: {params.get('RUN_TIMESTAMP', 'N/A')}\n\n")

        f.write("--- Dataset Information ---\n")
        f.write(f"Dataset Filename: {params.get('DATASET_FILENAME', 'N/A')}\n")
        f.write(f"Dataset Version: {params.get('DATASET_VERSION', 'N/A')}\n")
        f.write(f"H5 File Path: {params.get('H5_FILE_PATH', 'N/A')}\n")
        f.write(f"Train Samples: {params.get('TRAIN_SAMPLES', 'N/A')}\n")
        f.write(f"Validation Samples: {params.get('VAL_SAMPLES', 'N/A')}\n")
        f.write(f"Test Samples: {params.get('TEST_SAMPLES', 'N/A')}\n")
        f.write("\n")

        f.write("--- Model Architecture and Parameters ---\n")
        f.write(f"Input Shape: {params.get('INPUT_SHAPE', 'N/A')}\n")
        f.write(f"Number of Classes: {params.get('NUM_CLASSES', 'N/A')}\n")
        f.write(f"Total Model Parameters: {params.get('MODEL_TOTAL_PARAMS', 'N/A')}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Batch Size: {params.get('BATCH_SIZE', 'N/A')}\n")
        f.write(f"Epochs: {params.get('EPOCHS', 'N/A')}\n")
        f.write("\n")

        f.write("--- Training Results ---\n")
        f.write(f"Final Training Accuracy: {safe_float_format(params.get('FINAL_TRAINING_ACCURACY'))}\n")
        f.write(f"Final Validation Accuracy: {safe_float_format(params.get('FINAL_VALIDATION_ACCURACY'))}\n")
        f.write(f"Test Accuracy: {safe_float_format(params.get('TEST_ACCURACY'))}\n")
        f.write(f"Test Loss: {safe_float_format(params.get('TEST_LOSS'))}\n")

        if 'TOTAL_TRAINING_LOSS_FINAL' in params:
            f.write(f"Final Training Loss: {safe_float_format(params.get('TOTAL_TRAINING_LOSS_FINAL'))}\n")
            f.write(f"Final Validation Loss: {safe_float_format(params.get('TOTAL_VALIDATION_LOSS_FINAL'))}\n")

        if 'TEST_SEGMENTATION_ACCURACY_CLEAN' in params:
            f.write(f"Test Segmentation Accuracy: {safe_float_format(params.get('TEST_SEGMENTATION_ACCURACY_CLEAN'))}\n")
            f.write(f"Test Classification Accuracy: {safe_float_format(params.get('TEST_CLASSIFICATION_ACCURACY_CLEAN'))}\n")
            f.write(f"Test Segmentation Loss: {safe_float_format(params.get('TEST_SEGMENTATION_LOSS_CLEAN'))}\n")
            f.write(f"Test Classification Loss: {safe_float_format(params.get('TEST_CLASSIFICATION_LOSS_CLEAN'))}\n")
            f.write(f"Test Total Loss: {safe_float_format(params.get('TEST_TOTAL_LOSS_CLEAN'))}\n")

        f.write("\n")

        f.write("--- Paths ---\n")
        f.write(f"Model Save Path: {params.get('MODEL_SAVE_PATH', 'N/A')}\n")
        f.write(f"Results Directory: {params.get('RESULTS_DIRECTORY', 'N/A')}\n")