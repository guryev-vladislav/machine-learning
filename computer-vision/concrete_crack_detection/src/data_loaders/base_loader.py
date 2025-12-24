import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from torch.utils.data import Dataset
    from pathlib import Path
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

class BaseCrackDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.masks = []
        self.labels = []

    def _discover_files(self):
        raise NotImplementedError("Subclasses must implement _discover_files")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            mask_path = self.masks[idx]

            image = self._load_image(image_path)
            mask_or_label = self._load_mask(mask_path)

            if self.transform:
                if isinstance(mask_or_label, np.ndarray) and mask_or_label.ndim > 0:
                    transformed = self.transform(image=image, mask=mask_or_label)
                    image = transformed['image']
                    mask_or_label = transformed['mask']
                else:
                    transformed = self.transform(image=image)
                    image = transformed['image']

            return image, mask_or_label

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise

    def _load_image(self, image_path):
        raise NotImplementedError("Subclasses must implement _load_image")

    def _load_mask(self, mask_path):
        raise NotImplementedError("Subclasses must implement _load_mask")