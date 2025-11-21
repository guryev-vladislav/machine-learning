# main.py
import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_available_models():
    """Get list of available model versions"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    models = [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("model_v")]
    return sorted(models, key=lambda x: int(x.split('_v')[1]))


def print_menu():
    """Print main menu"""
    print("\n" + "=" * 50)
    print("    CONCRETE CRACK DETECTION SYSTEM")
    print("=" * 50)
    print("1. Train models (CNN classifier + U-Net segmentation)")
    print("2. Evaluate model")
    print("3. Run crack detection on image")
    print("4. Setup data structure")
    print("5. Exit")
    print("-" * 50)


def get_choice():
    """Get user choice with validation"""
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def interactive_mode():
    """Interactive mode with menu selection"""
    while True:
        print_menu()
        choice = get_choice()

        if choice == '1':
            interactive_train()
        elif choice == '2':
            interactive_evaluate()
        elif choice == '3':
            interactive_inference()
        elif choice == '4':
            setup_data()
        elif choice == '5':
            print("Goodbye!")
            break


def interactive_train():
    """Interactive model training for both CNN and U-Net"""
    print("\n--- MODEL TRAINING ---")

    # Training mode selection
    print("\nSelect training mode:")
    print("1. Full pipeline (CNN classifier + U-Net segmentation)")
    print("2. CNN classifier only (fast classification)")
    print("3. U-Net segmentation only (pixel-level detection)")

    mode_choice = input("Enter choice (1-3): ").strip()

    if mode_choice == '1':
        train_mode = 'full'
        print("Training both CNN classifier and U-Net segmentation model")
    elif mode_choice == '2':
        train_mode = 'classifier'
        print("Training CNN classifier only")
    elif mode_choice == '3':
        train_mode = 'segmentation'
        print("Training U-Net segmentation only")
    else:
        print("Invalid choice")
        return

    # Dataset selection
    print("\nSelect datasets for training:")
    print("1. DeepCrack (segmentation - 537 train + 237 test)")
    print("2. METU (segmentation - 1231 images)")
    print("3. SDNET2018 (classification - 56,000+ images)")
    print("4. All datasets")

    dataset_choice = input("Enter choice (1-4): ").strip()

    if dataset_choice == '1':
        datasets = ['deepcrack']
    elif dataset_choice == '2':
        datasets = ['metu']
    elif dataset_choice == '3':
        datasets = ['sdnet2018']
    elif dataset_choice == '4':
        datasets = ['deepcrack', 'metu', 'sdnet2018']
    else:
        print("Invalid choice")
        return

    # Only ask for epochs
    epochs = input("\nNumber of epochs [100]: ").strip() or "100"

    # Validate epochs
    try:
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print("Invalid number of epochs")
        return

    # Confirm and start training
    print(f"\n--- TRAINING SUMMARY ---")
    print(f"Mode: {train_mode}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Epochs: {epochs}")
    print("Batch size: 16 (from config)")
    print("Learning rate: 0.001 (from config)")

    confirm = input("\nStart training? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        args = type('Args', (), {
            'mode': 'train',
            'train_mode': train_mode,
            'datasets': datasets,
            'epochs': epochs
        })()

        train_models(args)
    else:
        print("Training cancelled")


def interactive_evaluate():
    """Interactive model evaluation"""
    print("\n--- MODEL EVALUATION ---")

    available_models = get_available_models()
    if not available_models:
        print("No trained models found. Please train a model first.")
        return

    # Model selection
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    try:
        model_choice = int(input(f"Select model (1-{len(available_models)}): "))
        model_version = available_models[model_choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice")
        return

    # Evaluation type
    print("\nSelect evaluation type:")
    print("1. Segmentation performance (IoU, Dice)")
    print("2. Classification performance (Accuracy, F1)")
    print("3. Full pipeline performance")

    eval_choice = input("Enter choice (1-3): ").strip()

    if eval_choice == '1':
        eval_type = 'segmentation'
        dataset = 'deepcrack'  # Default for segmentation
    elif eval_choice == '2':
        eval_type = 'classification'
        dataset = 'sdnet2018'  # Default for classification
    elif eval_choice == '3':
        eval_type = 'full'
        dataset = 'deepcrack'
    else:
        print("Invalid choice")
        return

    print(f"\n--- EVALUATION SUMMARY ---")
    print(f"Model: {model_version}")
    print(f"Evaluation type: {eval_type}")
    print(f"Dataset: {dataset}")

    confirm = input("\nStart evaluation? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        args = type('Args', (), {
            'mode': 'evaluate',
            'model_version': model_version.replace('model_', ''),
            'eval_type': eval_type,
            'dataset': dataset
        })()

        evaluate_model(args)
    else:
        print("Evaluation cancelled")


def interactive_inference():
    """Interactive crack detection on image"""
    print("\n--- CRACK DETECTION ON IMAGE ---")

    available_models = get_available_models()
    if not available_models:
        print("No trained models found. Please train a model first.")
        return

    # Model selection
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    try:
        model_choice = int(input(f"Select model (1-{len(available_models)}): "))
        model_version = available_models[model_choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice")
        return

    # Image path input
    image_path = input("\nPath to input image: ").strip()
    if not Path(image_path).exists():
        print("Image file not found")
        return

    # Detection mode
    print("\nSelect detection mode:")
    print("1. Fast classification only (crack/no crack)")
    print("2. Full pipeline (classification + segmentation)")
    print("3. Segmentation only (pixel-level detection)")

    detection_choice = input("Enter choice (1-3): ").strip()

    if detection_choice == '1':
        detection_mode = 'classification'
        output_path = "classification_result.png"
    elif detection_choice == '2':
        detection_mode = 'full'
        output_path = "detection_result.png"
    elif detection_choice == '3':
        detection_mode = 'segmentation'
        output_path = "segmentation_result.png"
    else:
        print("Invalid choice")
        return

    # Custom output path
    custom_output = input(f"Output path [{output_path}]: ").strip()
    if custom_output:
        output_path = custom_output

    # Threshold for segmentation
    if detection_mode in ['full', 'segmentation']:
        threshold = input("Segmentation threshold (0.1-0.9) [0.5]: ").strip() or "0.5"
        try:
            threshold = float(threshold)
            if not 0.1 <= threshold <= 0.9:
                raise ValueError
        except ValueError:
            print("Invalid threshold value")
            return
    else:
        threshold = 0.5

    print(f"\n--- DETECTION SUMMARY ---")
    print(f"Model: {model_version}")
    print(f"Input: {image_path}")
    print(f"Output: {output_path}")
    print(f"Mode: {detection_mode}")
    if detection_mode in ['full', 'segmentation']:
        print(f"Threshold: {threshold}")

    confirm = input("\nRun detection? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        args = type('Args', (), {
            'mode': 'inference',
            'model_version': model_version.replace('model_', ''),
            'image': image_path,
            'output': output_path,
            'detection_mode': detection_mode,
            'threshold': threshold
        })()

        run_detection(args)
    else:
        print("Detection cancelled")


def setup_data():
    """Setup dataset structure"""
    from scripts.setup_data import main as setup_main
    print("\nSetting up data structure...")
    setup_main()


def train_models(args):
    """Train models based on arguments"""
    try:
        # Update config with epochs only
        import yaml
        with open('configs/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        config['training']['epochs'] = args.epochs

        with open('configs/training_config.yaml', 'w') as f:
            yaml.dump(config, f)

        # Train based on mode
        if args.train_mode in ['full', 'classifier']:
            print("Training CNN classifier...")
            from scripts.train_classification import main as train_cls_main
            train_cls_main()

        if args.train_mode in ['full', 'segmentation']:
            print("Training U-Net segmentation...")
            from scripts.train_segmentation import main as train_seg_main
            train_seg_main()

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def evaluate_model(args):
    """Evaluate model based on arguments"""
    from scripts.evaluate import main as eval_main

    # Pass arguments directly
    import sys
    old_argv = sys.argv
    sys.argv = [
        'evaluate.py',
        '--model-version', args.model_version,
        '--dataset', args.dataset,
        '--eval-type', args.eval_type
    ]

    try:
        eval_main()
    finally:
        sys.argv = old_argv


def run_detection(args):
    """Run crack detection on single image"""
    from scripts.inference_demo import main as inference_main

    # Pass arguments directly
    import sys
    old_argv = sys.argv
    sys.argv = [
        'inference_demo.py',
        '--model-version', args.model_version,
        '--image', args.image,
        '--output', args.output,
        '--detection-mode', args.detection_mode,
        '--threshold', str(args.threshold)
    ]

    try:
        inference_main()
    finally:
        sys.argv = old_argv


def main():
    setup_logging()

    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(description='Concrete Crack Detection System')
        parser.add_argument('--mode', type=str, required=True,
                            choices=['train', 'evaluate', 'inference', 'setup'])
        parser.add_argument('--train-mode', type=str, choices=['full', 'classifier', 'segmentation'])
        parser.add_argument('--datasets', type=str, nargs='+')
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--model-version', type=str)
        parser.add_argument('--eval-type', type=str)
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--image', type=str)
        parser.add_argument('--output', type=str)
        parser.add_argument('--detection-mode', type=str)
        parser.add_argument('--threshold', type=float, default=0.5)

        args = parser.parse_args()

        if args.mode == 'train':
            train_models(args)
        elif args.mode == 'evaluate':
            evaluate_model(args)
        elif args.mode == 'inference':
            run_detection(args)
        elif args.mode == 'setup':
            setup_data()
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()