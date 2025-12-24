import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice_score = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice_score

class CombinedLoss(nn.Module):
    def __init__(self, weight=2.0, pos_weight_value=25.0):
        super().__init__()
        self.dice_weight = weight
        self.pos_weight_value = pos_weight_value

    def forward(self, pred, target):
        device = pred.device
        pos_weight = torch.tensor([self.pos_weight_value], device=device)

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        pred_sig = torch.sigmoid(pred)
        d_loss = dice_loss(pred_sig, target, smooth=1.0)

        return bce_loss + self.dice_weight * d_loss