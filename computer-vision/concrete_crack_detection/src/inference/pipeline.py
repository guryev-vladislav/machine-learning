# src/inference/pipeline.py
import torch
import logging
import json
from pathlib import Path
from .predictor import CrackPredictor
from ..utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, model_path, model_type="unet"):
        self.predictor = CrackPredictor(model_path, model_type)
        self.metrics = SegmentationMetrics()

    def evaluate_dataset(self, dataset):
        try:
            self.metrics.reset()
            all_predictions = []

            for i, (image, mask) in enumerate(dataset):
                image_tensor = image.unsqueeze(0).to(self.predictor.device)

                with torch.no_grad():
                    prediction = self.predictor.model(image_tensor)
                    prediction = torch.sigmoid(prediction)

                self.metrics.update(prediction, mask.unsqueeze(0))
                all_predictions.append(prediction.cpu().numpy())

                if i % 50 == 0:
                    logger.info(f"Processed {i}/{len(dataset)} samples")

            final_metrics = self.metrics.get_all_metrics()
            logger.info(f"Evaluation completed. IoU: {final_metrics['iou']:.4f}")

            return final_metrics, all_predictions

        except Exception as e:
            logger.error(f"Dataset evaluation failed: {e}")
            raise

    @staticmethod
    def generate_report(metrics, result_path):
        try:
            report_path = Path(result_path) / "test_results" / "final_metrics.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise