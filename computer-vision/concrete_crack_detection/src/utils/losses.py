import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Dice Loss (вспомогательная функция) ---
def dice_loss(pred, target, smooth=1.0):
    """
    Рассчитывает Dice Loss. Smooth=1.0 помогает, когда трещин на картинке мало.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice_score = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice_score


# --- 2. Combined Loss (BCE + Dice) ---
class CombinedLoss(nn.Module):
    def __init__(self, weight=2.0, pos_weight_value=25.0):
        """
        weight: вес для Dice (увеличиваем до 2.0, чтобы форма была важнее фона)
        pos_weight_value: вес пикселя трещины (поднимаем до 25.0 для супер-чувствительности)
        """
        super().__init__()
        self.dice_weight = weight
        self.pos_weight_value = pos_weight_value

    def forward(self, pred, target):
        # Автоматическое определение устройства (решает проблему CUDA vs CPU)
        device = pred.device
        pos_weight = torch.tensor([self.pos_weight_value], device=device)

        # 1. Weighted BCE: заставляет сеть "кричать" при виде любого намека на трещину
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

        # 2. Dice Loss: заставляет сеть собирать пиксели в единые линии
        pred_sig = torch.sigmoid(pred)

        # Используем встроенную функцию для стабильности
        d_loss = dice_loss(pred_sig, target, smooth=1.0)

        # Суммируем: BCE ищет точки, Dice соединяет их в трещины
        return bce_loss + self.dice_weight * d_loss