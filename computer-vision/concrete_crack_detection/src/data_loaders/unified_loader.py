import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from pathlib import Path
    import random
    from .base_loader import BaseCrackDataset
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)


class UnifiedCrackDataset(BaseCrackDataset):
    def __init__(self, crack_dir, non_crack_dir, is_train=True, target_size=(256, 256), task='segmentation',
                 unet_model=None):
        super().__init__(crack_dir, is_train=is_train)
        self.target_size = target_size
        self.task = task
        self.unet_model = unet_model
        self.crack_dir = Path(crack_dir).resolve()
        self.non_crack_dir = Path(non_crack_dir).resolve()

        if self.is_train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=target_size, scale=(0.5, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        self._discover_files()

    def _discover_files(self):
        crack_images_dir = self.crack_dir / "rgb"
        crack_masks_dir = self.crack_dir / "BW"

        logger.info(f"Checking images in: {crack_images_dir}")

        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        all_imgs = []
        for e in exts:
            all_imgs.extend(list(crack_images_dir.glob(e)))

        self.images, self.masks, self.labels = [], [], []

        for img_path in sorted(all_imgs):
            mask_path = crack_masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = crack_masks_dir / img_path.name

            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)
                self.labels.append(1.0)

        if self.task == 'classification':
            non_crack_imgs = []
            for obj in ["Decks", "Pavements", "Walls"]:
                nc_dir = self.non_crack_dir / obj / "Non-cracked"
                if nc_dir.exists():
                    for e in exts:
                        non_crack_imgs.extend(list(nc_dir.glob(e)))

            if non_crack_imgs:
                random.seed(42)
                num_to_select = min(len(non_crack_imgs), len(self.images))
                selected = random.sample(non_crack_imgs, num_to_select)
                for path in selected:
                    self.images.append(path)
                    self.masks.append(None)
                    self.labels.append(0.0)

        logger.info(f"Total images found: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = str(self.images[idx])
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            logger.error(f"Failed to load image: {image_path}")
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        if self.masks[idx]:
            mask = (cv2.imread(str(self.masks[idx]), 0) > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        transformed = self.aug(image=image, mask=mask)
        img_t, mask_t = transformed['image'], transformed['mask']

        if self.task == 'classification':
            if self.unet_model is not None:
                self.unet_model.eval()
                with torch.no_grad():
                    dev = next(self.unet_model.parameters()).device
                    pred = torch.sigmoid(self.unet_model(img_t.unsqueeze(0).to(dev)))
                    final_input = (pred > 0.5).float().squeeze(0).repeat(3, 1, 1)
            else:
                final_input = mask_t.unsqueeze(0).repeat(3, 1, 1)
        else:
            final_input = img_t

        return final_input, mask_t.unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)