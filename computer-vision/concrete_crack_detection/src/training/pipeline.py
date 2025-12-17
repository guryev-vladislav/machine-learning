import torch
import logging
import datetime
import shutil
import json
import sys
import cv2
from pathlib import Path

# Настройка путей
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Безопасные импорты вынесены из функций
try:
    from src.training.trainer import Trainer
    from src.utils.config import Config
    from src.utils.visualizer import save_learning_curves, save_detailed_inference
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
except ImportError as e:
    logger.error(f"Import error in pipeline.py: {e}")
    sys.exit(1)


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        # Фиксация timestamp для использования во всех методах (устраняет Unresolved reference)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path("outputs") / f"RUN_{self.timestamp}"
        self.plots_dir = self.run_dir / "plots"
        self.weights_dir = self.run_dir / "weights"

        for d in [self.plots_dir, self.weights_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.unet = None
        self.classifier = None
        self._backup_config()

    def _backup_config(self):
        cfg_path = Path("configs/train_config.yaml")
        if cfg_path.exists():
            shutil.copy(cfg_path, self.run_dir / "train_config.yaml")

    def train_segmentation(self):
        logger.info("Stage 1: UNet Segmentation Training")
        trainer = Trainer(task='segmentation')
        history, weights = trainer.train()
        torch.save(weights, self.weights_dir / "unet_final.pth")
        save_learning_curves(history, self.run_dir, "UNet_Segmentation")
        return history

    def train_classification(self):
        logger.info("Stage 2: CNN Classification Training")
        trainer = Trainer(task='classification')
        history, weights = trainer.train()
        torch.save(weights, self.weights_dir / "classifier_final.pth")
        save_learning_curves(history, self.run_dir, "CNN_Classification")
        return history

    def run_inference(self, test_image_path):
        """Создает комплексную научную панель анализа"""
        if not test_image_path or not Path(test_image_path).exists():
            logger.error(f"Test image not found at: {test_image_path}")
            return

        # Инициализация моделей (веса берутся из текущей сессии обучения)
        if self.unet is None:
            self.unet = UNet(n_channels=3, n_classes=1).to(self.device)
            self.unet.load_state_dict(torch.load(self.weights_dir / "unet_final.pth"))
        if self.classifier is None:
            self.classifier = SimpleCNN(n_channels=3, n_classes=1).to(self.device)
            self.classifier.load_state_dict(torch.load(self.weights_dir / "classifier_final.pth"))

        self.unet.eval()
        self.classifier.eval()

        # Загрузка и препроцессинг
        img = cv2.imread(str(test_image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sz = self.config.PATCH_SIZE
        img_res = cv2.resize(img_rgb, (sz, sz))

        tensor = torch.from_numpy(img_res.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            mask = torch.sigmoid(self.unet(tensor)).cpu().numpy()[0, 0]
            prob = torch.sigmoid(self.classifier(tensor)).item()

        # Гибридная логика вердикта
        is_cracked = prob > 0.5 or (mask > 0.5).sum() > 100

        save_detailed_inference(img_res, mask, prob, is_cracked, self.plots_dir, Path(test_image_path).stem)
        logger.info(f"Scientific report generated for {Path(test_image_path).name}")

    def run_full_experiment(self, test_image=None):
        h_seg = self.train_segmentation()
        h_cls = self.train_classification()

        if test_image:
            self.run_inference(test_image)

        summary = {
            "date": self.timestamp,
            "metrics": {
                "unet_final_iou": h_seg['metric'][-1],
                "cnn_final_acc": h_cls['metric'][-1]
            },
            "config_snapshot": self.config.data['training']
        }
        with open(self.run_dir / "summary.json", "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info(f"Experiment cycle complete. Results in {self.run_dir}")