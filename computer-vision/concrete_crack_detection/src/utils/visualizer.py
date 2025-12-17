import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_learning_curves(history, save_path, task_name):
    """
    Создает графики Loss и Metric для конкретной задачи (UNet или CNN).
    Используется стандарт оформления для научных публикаций.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    metric_name = 'IoU' if 'segmentation' in task_name.lower() else 'Accuracy'

    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Training Progress: {task_name}", fontsize=16)

    # Левый график: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', linewidth=2, label='Training Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Loss Convergence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Правый график: Metric
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['metric'], 'g-s', linewidth=2, label=f'Training {metric_name}')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} Improvement')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f"learning_curves_{task_name.lower()}.png", dpi=300)
    plt.close()


def save_detailed_inference(image, mask, prob, is_cracked, save_path, name):
    """
    Сохраняет детальный результат инференса:
    Оригинал | Тепловая карта UNet | Финальный вердикт
    """
    plt.figure(figsize=(15, 5))

    # 1. Исходное изображение
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    # 2. Сегментация (Heatmap)
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    mask_colored = np.ma.masked_where(mask < 0.1, mask)
    plt.imshow(mask_colored, alpha=0.7, cmap='jet')
    plt.title("Segmentation (UNet)")
    plt.axis('off')

    # 3. Уверенность классификатора
    plt.subplot(1, 3, 3)
    color = 'red' if is_cracked else 'green'
    status = "CRACK DETECTED" if is_cracked else "CLEAN"

    plt.bar(["Confidence"], [prob], color=color, alpha=0.6)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold 0.5')
    plt.ylim(0, 1.1)
    plt.title(f"Decision: {status}\nScore: {prob:.4f}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path / f"inference_report_{name}.png", dpi=300)
    plt.close()

    def save_detailed_inference(image, mask, prob, is_cracked, save_path, name):
        """
        Создает комплексный отчет: Оригинал | Тепловая карта трещины | Вердикт CNN.
        Используется строгий стиль оформления без лишних элементов.
        """
        plt.figure(figsize=(18, 6))

        # Панель 1: Исходное изображение
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Surface Inspection", fontsize=14, pad=15)
        plt.axis('off')

        # Панель 2: Результат UNet (Segmentation Map)
        plt.subplot(1, 3, 2)
        plt.imshow(image)
        # Наложение маски: используем jet или hot для выделения интенсивности
        mask_overlay = np.ma.masked_where(mask < 0.1, mask)
        plt.imshow(mask_overlay, alpha=0.7, cmap='jet', vmin=0, vmax=1)
        plt.title("UNet Segmentation Overlay", fontsize=14, pad=15)
        plt.axis('off')

        # Панель 3: Анализ уверенности и финальный вердикт
        plt.subplot(1, 3, 3)
        color = 'firebrick' if is_cracked else 'forestgreen'
        verdict_text = "STATUS: DEFECT DETECTED" if is_cracked else "STATUS: CLEAR"

        # Рисуем шкалу уверенности
        bars = plt.bar(["Probability Score"], [prob], color=color, alpha=0.8, width=0.5)
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.5)')

        # Добавляем текстовый вердикт под заголовком
        plt.ylim(0, 1.1)
        plt.ylabel("Confidence Level", fontsize=12)
        plt.title(f"{verdict_text}\n(CNN Output: {prob:.4f})", fontsize=14, pad=15, color=color, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle=':', alpha=0.5)

        plt.tight_layout()

        # Сохранение в высоком качестве для отчетов
        file_name = f"inference_report_{name}.png"
        plt.savefig(save_path / file_name, dpi=300, bbox_inches='tight')
        plt.close()