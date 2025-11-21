# scripts/inference_demo.py
import argparse
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.inference.predictor import CrackPredictor
except ImportError as e:
    print(f"Import error: {e}")
    raise


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', type=str, required=True, help='Model version to use')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='prediction_result.png', help='Output path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    return parser.parse_args()


def visualize_prediction(original_image, prediction, output_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(prediction, cmap='gray')
    ax2.set_title('Crack Prediction')
    ax2.axis('off')

    overlay = original_image.copy()
    mask = prediction > 0.5
    overlay[mask] = [255, 0, 0]
    ax3.imshow(overlay)
    ax3.set_title('Overlay')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    try:
        model_path = Path("models") / args.model_version / "model.pth"
        if not model_path.exists():
            raise ValueError(f"Model {args.model_version} not found")

        predictor = CrackPredictor(model_path)

        original_image = cv2.imread(args.image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        prediction = predictor.predict_image(args.image, args.threshold)

        visualize_prediction(original_image, prediction, args.output)

        crack_pixels = np.sum(prediction > args.threshold)
        total_pixels = prediction.size
        crack_ratio = crack_pixels / total_pixels

        logger.info(f"Prediction completed")
        logger.info(f"Crack area ratio: {crack_ratio:.4f}")
        logger.info(f"Result saved to {args.output}")

    except Exception as e:
        logger.error(f"Inference demo failed: {e}")
        raise


if __name__ == "__main__":
    main()