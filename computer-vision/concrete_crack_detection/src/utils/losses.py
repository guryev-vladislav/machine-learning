import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Dice Loss ---
def dice_loss(pred, target, smooth=1e-6):
    """
    Рассчитывает Dice Loss для бинарной сегментации.

    Args:
        pred (Tensor): Предсказания модели (логиты после Sigmoid).
        target (Tensor): Истинные маски (0 или 1).
    """
    # Преобразование в одномерный вектор для простоты расчета
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Intersection и Union
    intersection = (pred * target).sum()
    dice_score = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # Dice Loss = 1 - Dice Score
    return 1 - dice_score


# --- 2. Combined Loss (BCE + Dice) ---
class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь:
    Binary Cross-Entropy (BCE) + Dice Loss.
    """

    def __init__(self, weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Вес для Dice Loss (можно настраивать, 0.5 - стандартный старт)
        self.dice_weight = weight

    def forward(self, pred_logits, target):
        # 1. BCE Loss: работает с логитами (до Sigmoid)
        bce = self.bce_loss(pred_logits, target)

        # 2. Dice Loss: требует вероятностей (после Sigmoid)
        pred_probs = torch.sigmoid(pred_logits)
        dice = dice_loss(pred_probs, target)

        # Комбинируем потери
        total_loss = bce + self.dice_weight * dice

        return total_loss