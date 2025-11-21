# src/inference/predictor.py
import torch
import logging
import cv2
import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2
from ..models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class CrackPredictor:
    def __init__(self, model_path, model_type="unet", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModelFactory.load_model(model_path, model_type)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_path, threshold=0.5):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

            transform = albumentations.Compose([
                albumentations.Resize(512, 512),
                albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(image_tensor)
                prediction = torch.sigmoid(prediction)
                prediction = (prediction > threshold).float()

            return prediction.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {e}")
            raise

    def predict_batch(self, image_paths, threshold=0.5):
        predictions = []
        for path in image_paths:
            pred = self.predict_image(path, threshold)
            predictions.append(pred)
        return predictions