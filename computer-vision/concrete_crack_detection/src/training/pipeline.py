import sys
import logging
from pathlib import Path
import torch
import cv2
import numpy as np
from datetime import datetime

# Настройка путей
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.training.trainer import Trainer
from src.utils.visualizer import save_training_history, save_multiscale_comparison

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE

        # --- Создаем структуру папок OUTPUTS ---
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_run_dir = Path(project_root) / "outputs" / f"run_{run_id}"
        self.checkpoints_dir = self.base_run_dir / "models"
        self.plots_dir = self.base_run_dir / "plots"

        for d in [self.checkpoints_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"All data for this run will be saved in: {self.base_run_dir}")

    def train_segmentation(self):
        logger.info("Stage 1: UNet Segmentation Training")
        trainer = Trainer(task='segmentation')
        history = trainer.train()

        # Сохраняем модель в папку запуска
        trainer.save_checkpoint(self.checkpoints_dir / "unet_final.pth")
        save_training_history(history, "unet_training", self.plots_dir)
        return history

    def train_classification(self):
        logger.info("Stage 2: CNN Classification Training")
        trainer = Trainer(task='classification')
        history = trainer.train()

        # Сохраняем модель в папку запуска
        trainer.save_checkpoint(self.checkpoints_dir / "classifier_final.pth")
        save_training_history(history, "cnn_training", self.plots_dir)
        return history

    def run_multiscale_test(self, test_image_path):
        """Тестирование на разных масштабах с сохранением исходника"""
        logger.info(f"Running multiscale test for: {test_image_path}")

        # Проверка существования файла
        if not Path(test_image_path).exists():
            logger.error(f"File not found: {test_image_path}")
            return

        from src.models.unet import UNet
        from src.models.classifier import SimpleCNN

        unet = UNet(n_channels=3, n_classes=1).to(self.device)
        classifier = SimpleCNN(n_channels=3, n_classes=1).to(self.device)

        unet.load_state_dict(torch.load(self.checkpoints_dir / "unet_final.pth"))
        classifier.load_state_dict(torch.load(self.checkpoints_dir / "classifier_final.pth"))

        unet.eval()
        classifier.eval()

        img = cv2.imread(str(test_image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        scales = {"Original": 256, "Medium": 128, "Small": 64}
        results = []

        for name, size in scales.items():
            img_res = cv2.resize(img_rgb, (size, size))
            img_input = cv2.resize(img_res, (self.config.PATCH_SIZE, self.config.PATCH_SIZE))

            tensor = torch.from_numpy(img_input.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
            # Нормализация
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            tensor = (tensor - mean) / std

            with torch.no_grad():
                mask = torch.sigmoid(unet(tensor)).cpu().numpy()[0, 0]
                prob = torch.sigmoid(classifier(tensor)).item()

            results.append({
                'image': img_input,
                'mask': mask,
                'prob': prob,
                'scale_name': name
            })

        # Передаем исходное изображение img_rgb отдельно для визуализации
        save_multiscale_comparison(results, self.plots_dir, Path(test_image_path).stem, original_img=img_rgb)

    def run_full_experiment(self, test_image=None):
        self.train_segmentation()
        self.train_classification()
        if test_image:
            self.run_multiscale_test(test_image)