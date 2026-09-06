import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from src.utils.config import Config

class YOLOCrackDetector:
    def __init__(self, config=None, model_path=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.conf_threshold = self.config.CONF_THRESHOLD
        self.model_path = model_path
        self.model = None

    def _init_model(self):
        if self.model is None:
            try:
                from ultralytics import YOLO
                if self.model_path and Path(self.model_path).exists():
                    model_src = Path(self.model_path)
                    logger.info(f"Loading model from {model_src}")
                    self.model = YOLO(str(model_src))
                    self.model.to(self.device)
                    return

                cfg_model = getattr(self.config, 'MODEL_NAME', None)
                if cfg_model:
                    cfg_model = str(cfg_model)
                    cfg_path = Path(cfg_model)
                    if not cfg_path.is_absolute():
                        cfg_path = Path(self.config.PROJECT_PATH) / cfg_path
                    pretrained_dir = Path(self.config.PROJECT_PATH) / 'src' / 'models' / 'pretrained'
                    pretrained_candidate = pretrained_dir / Path(cfg_model).name

                    if cfg_path.exists():
                        logger.info(f"Loading model from config path: {cfg_path}")
                        self.model = YOLO(str(cfg_path))
                        self.model.to(self.device)
                        return
                    elif pretrained_candidate.exists():
                        logger.info(f"Loading model from pretrained dir: {pretrained_candidate}")
                        self.model = YOLO(str(pretrained_candidate))
                        self.model.to(self.device)
                        return

                logger.info(f"Loading pretrained model: {self.config.MODEL_NAME}")
                self.model = YOLO(self.config.MODEL_NAME)
                self.model.to(self.device)
            except ImportError as e:
                logger.error(f"Failed to load ultralytics: {e}")
                raise

    def train(self, dataset_root=None, epochs=None, batch_size=None, save_period: int = 0):
        self._init_model()

        dataset_root = dataset_root or self.config.DATASET_ROOT
        epochs = epochs or self.config.EPOCHS
        batch_size = batch_size or self.config.BATCH_SIZE

        logger.info(f"Starting training...")
        logger.info(f"  Dataset root: {dataset_root}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")

        train_kwargs = dict(
            data=str(dataset_root),
            epochs=epochs,
            imgsz=self.config.IMGSZ,
            batch=batch_size,
            device=self.device,
            patience=self.config.PATIENCE,
            workers=self.config.NUM_WORKERS,
            save=True,
            task=self.config.TASK,
        )
        if save_period and int(save_period) > 0:
            train_kwargs['save_period'] = int(save_period)

        results = self.model.train(**train_kwargs)

        logger.info("Training completed!")
        return results

    def predict(self, source, conf=None):
        self._init_model()

        conf = conf or self.conf_threshold

        logger.info(f"Running inference on: {source}")
        results = self.model.predict(
            source=source,
            conf=conf,
            device=self.device,
            verbose=False
        )

        return results

    def classify_frame(self, frame):
        self._init_model()
        results = self.model(frame)

        if not results:
            return None

        result = results[0]

        top_name = None
        top_conf = 0.0

        probs = getattr(result, 'probs', None)
        if probs is not None:
            if hasattr(probs, 'top1'):
                try:
                    top_idx = int(probs.top1)
                    top_name = result.names.get(top_idx, str(top_idx))
                    try:
                        top_conf = float(probs.top1conf.item())
                    except Exception:
                        try:
                            top_conf = float(probs.top1conf)
                        except Exception:
                            top_conf = 0.0
                except Exception:
                    top_name = None
            else:
                try:
                    probs_arr = probs.numpy()
                except Exception:
                    try:
                        probs_arr = probs.cpu().numpy()
                    except Exception:
                        probs_arr = None

                if probs_arr is not None:
                    probs_arr = probs_arr.ravel()
                    top_idx = int(probs_arr.argmax())
                    top_conf = float(probs_arr[top_idx])
                    top_name = result.names.get(top_idx, str(top_idx))

        if top_name is None:
            if hasattr(result, 'top1'):
                try:
                    top_idx = int(result.top1)
                    top_name = result.names.get(top_idx, str(top_idx))
                except Exception:
                    top_name = None
            if hasattr(result, 'probs') and hasattr(result.probs, 'top1conf') and top_name is None:
                try:
                    top_conf = float(result.probs.top1conf.item())
                except Exception:
                    pass

        if top_name is None:
            try:
                if result.names:
                    top_name = list(result.names.values())[0]
            except Exception:
                top_name = 'unknown'

        return {
            'class': top_name,
            'confidence': top_conf,
            'is_crack': top_name == 'crack'
        }

    def save_model(self, path):
        self._init_model()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {path}")
        self.model.save(str(path))

    def export_model(self, format='onnx', path=None):
        self._init_model()

        if path is None:
            path = self.config.OUTPUTS_PATH / f"model.{format}"

        logger.info(f"Exporting model to {format}: {path}")
        self.model.export(format=format, imgsz=self.config.IMGSZ)
        logger.info(f"Model exported successfully!")









