# src/models/classifier.py
import logging

import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

class CrackClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        try:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            raise

    def forward(self, x):
        return self.backbone(x)