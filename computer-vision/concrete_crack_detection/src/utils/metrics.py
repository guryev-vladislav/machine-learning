import torch


def calculate_metrics(pred, target, threshold=0.5):
    """Вычисляет IoU и Dice для бинарной сегментации"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

    return iou.item(), dice.item()