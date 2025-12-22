import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_training_history(history, task_name, save_dir):
    """
    Сохраняет графики обучения (Loss и Metric).
    """
    if not isinstance(history, dict) or 'train_loss' not in history:
        print(f"Error: history format is incorrect for {task_name}")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Training Progress: {task_name}", fontsize=14)

    # График Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-o', label='Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График Метрики
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['metric'], 'g-s', label='Metric')
    plt.title("Metric Score")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = Path(save_dir) / f"{task_name}_curves.png"
    plt.savefig(save_path)
    plt.close()
    print(f"History plot saved to {save_path}")


def save_multiscale_comparison(results, save_path, filename, original_img=None):
    """
    Создает детальный отчет:
    Строки = Разные масштабы изображения.
    Колонки = [Входное фото] | [Результат UNet] | [Вердикт CNN]
    """
    n = len(results)
    # Создаем сетку: n строк (по количеству масштабов), 3 колонки
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    plt.suptitle(f"Detection Analysis Report: {filename}", fontsize=18, fontweight='bold')

    # Если n=1, matplotlib возвращает одномерный массив осей, превращаем в 2D для удобства
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(results):
        # --- Колонка 1: Входное изображение (то, что видит модель) ---
        ax_input = axes[i, 0]
        ax_input.imshow(res['image'])
        ax_input.set_title(f"Input ({res['scale_name']})", fontsize=12)
        ax_input.axis('off')

        # --- Колонка 2: Результат сегментации UNet ---
        ax_unet = axes[i, 1]
        ax_unet.imshow(res['image'])
        # Накладываем маску. Порог 0.05, чтобы видеть "неуверенные" предсказания
        mask_overlay = np.ma.masked_where(res['mask'] < 0.05, res['mask'])
        ax_unet.imshow(mask_overlay, alpha=0.6, cmap='jet')
        ax_unet.set_title("UNet Segmentation Overlay", fontsize=12)
        ax_unet.axis('off')

        # --- Колонка 3: Вердикт классификатора CNN ---
        ax_cnn = axes[i, 2]
        prob = res['prob']
        color = 'red' if prob > 0.5 else 'green'
        # Рисуем шкалу уверенности
        ax_cnn.barh(["Crack Prob"], [prob], color=color, height=0.4)
        ax_cnn.set_xlim(0, 1)
        ax_cnn.axvline(x=0.5, color='black', linestyle='--')
        ax_cnn.set_title(f"CNN Confidence: {prob:.4f}", fontsize=12)

        # Стилизация графика классификации
        ax_cnn.spines['top'].set_visible(False)
        ax_cnn.spines['right'].set_visible(False)
        ax_cnn.set_xlabel("Probability")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    final_path = Path(save_path) / f"{filename}_detailed_report.png"
    plt.savefig(final_path, dpi=150)
    plt.close()
    print(f"Detailed multiscale report saved to: {final_path}")


def save_detailed_inference(image, mask, prob, is_cracked, save_dir, filename):
    """
    Сохраняет упрощенный результат (если нужно для одиночных тестов).
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    mask_visible = np.ma.masked_where(mask < 0.05, mask)
    plt.imshow(mask_visible, alpha=0.5, cmap='hot')
    plt.title(f"UNet Mask")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(["CNN Score"], [prob], color='red' if is_cracked else 'green')
    plt.ylim(0, 1)
    plt.title(f"Verdict: {'CRACK' if is_cracked else 'CLEAN'}")

    plt.savefig(Path(save_dir) / f"inf_{filename}.png")
    plt.close()