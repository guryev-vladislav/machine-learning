import os
import sys
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import cv2
    import numpy as np
    import yaml
    from src.models.unet import UNet
    from src.models.classifier import SimpleCNN
    from src.utils.config import Config
    from src.training.trainer import Trainer
    from src.utils.visualizer import save_multiscale_comparison, plot_training_history
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

class TrainingPipeline:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config.BASE_PATH) / "outputs" / f"run_{self.run_id}"
        self.checkpoints_dir = self.run_dir / "models"
        self.plots_dir = self.run_dir / "plots"

        for d in [self.checkpoints_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._export_config()

    def _export_config(self):
        dest = self.run_dir / "config_used.yaml"
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(self.config).items() if not k.startswith('__')}
        with open(dest, 'w') as f:
            yaml.dump(config_dict, f)

    def run_full_experiment(self, test_images_paths=None):
        logger.info(f"Starting experiment: {self.run_id}")

        trainer_seg = Trainer(task='segmentation', save_dir=self.checkpoints_dir)
        unet_model, history_seg = trainer_seg.train()
        plot_training_history(history_seg, self.plots_dir / "unet_curves.png", "UNet")

        trainer_cls = Trainer(task='classification', save_dir=self.checkpoints_dir, unet_model=unet_model)
        cnn_model, history_cls = trainer_cls.train()
        plot_training_history(history_cls, self.plots_dir / "cnn_curves.png", "CNN")

        if test_images_paths:
            for img_path in test_images_paths:
                self._run_visual_test(img_path, unet_model, cnn_model)

    def _run_visual_test(self, img_path, unet, cnn):
        if not Path(img_path).exists():
            return
        unet.eval()
        cnn.eval()

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        scales = {"256px": 256, "128px": 128, "64px": 64}
        results = []

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        for name, size in scales.items():
            img_low = cv2.resize(img_rgb, (size, size))
            img_input = cv2.resize(img_low, (self.config.PATCH_SIZE, self.config.PATCH_SIZE))

            t = torch.from_numpy(img_input).permute(2, 0, 1).float().to(self.device) / 255.0
            t = (t.unsqueeze(0) - mean) / std

            with torch.no_grad():
                mask_prob = torch.sigmoid(unet(t))
                cnn_in = (mask_prob > 0.5).float().repeat(1, 3, 1, 1)
                prob = torch.sigmoid(cnn(cnn_in)).item()
                results.append((name, mask_prob.squeeze().cpu().numpy(), prob, img_input))

        save_multiscale_comparison(results, self.plots_dir / f"result_{Path(img_path).stem}.png")