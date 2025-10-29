# utils/metrics.py
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
        f.write(f"Model Type: {params.get('MODEL_TYPE', 'N/A')}\n\n")

        f.write("--- Training Results ---\n")
        f.write(f"Test Accuracy: {safe_float_format(params.get('TEST_ACCURACY'))}\n")
        f.write(f"Test Loss: {safe_float_format(params.get('TEST_LOSS'))}\n")

        if 'FINAL_TRAINING_ACCURACY' in params:
            f.write(f"Final Training Accuracy: {safe_float_format(params.get('FINAL_TRAINING_ACCURACY'))}\n")
            f.write(f"Final Validation Accuracy: {safe_float_format(params.get('FINAL_VALIDATION_ACCURACY'))}\n")