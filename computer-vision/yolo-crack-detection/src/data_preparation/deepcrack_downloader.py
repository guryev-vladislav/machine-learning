"""DeepCrack dataset downloader from Kaggle."""

import sys
from pathlib import Path

# Add project root to PYTHONPATH for standalone execution
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import kagglehub

from src.data_preparation.data_downloader import DataDownloader
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

DEEPCRACK_KAGGLE_ID = "rukiyeaydn/deepcrack-dataset"


class DeepCrackDownloader(DataDownloader):
    """Download DeepCrack concrete crack dataset from Kaggle."""

    def __init__(self, download_path: Path):
        self._download_path = download_path

    @property
    def dataset_name(self) -> str:
        return "deepcrack"

    @property
    def download_path(self) -> Path:
        return self._download_path

    def download(self) -> Path:
        """Download DeepCrack from Kaggle and organize into expected structure.

        Expected Kaggle structure:
            DeepCrack/
                train_img/
                train_lab/
                test_img/
                test_lab/

        Returns:
            Path to the downloaded dataset.
        """
        kaggle_path = kagglehub.dataset_download(DEEPCRACK_KAGGLE_ID)
        kaggle_path = Path(kaggle_path)
        self._download_path.mkdir(parents=True, exist_ok=True)
        self._organize_deepcrack(kaggle_path)
        logger.info(f"DeepCrack downloaded and organized to {self._download_path}")
        return self._download_path

    def _organize_deepcrack(self, kaggle_path: Path):
        """Organize Kaggle DeepCrack structure into expected directories."""
        expected_dirs = ["train_img", "train_lab", "test_img", "test_lab"]

        for dir_name in expected_dirs:
            src = kaggle_path / dir_name
            if not src.exists():
                logger.warning(f"Directory not found in Kaggle dataset: {dir_name}")
                continue

            dest = self._download_path / dir_name
            dest.mkdir(parents=True, exist_ok=True)

            # Copy files, skip existing
            files = list(src.glob("*"))
            copied = 0
            for file_path in files:
                if file_path.is_file():
                    dest_file = dest / file_path.name
                    if not dest_file.exists():
                        import shutil
                        shutil.copy2(file_path, dest_file)
                        copied += 1

            logger.info(f"  Copied {copied} files from {dir_name}")

    def _has_files(self) -> bool:
        """Check if DeepCrack dataset has files in expected directories."""
        expected_dirs = ["train_img", "train_lab", "test_img", "test_lab"]

        for dir_name in expected_dirs:
            dir_path = self._download_path / dir_name
            if not dir_path.exists():
                return False
            if not any(dir_path.iterdir()):
                return False

        return True


if __name__ == '__main__':
    from src.utils.logging_setup import shutdown_logging

    download_path = Path(__file__).resolve().parents[2] / 'data' / 'external' / 'deepcrack'
    downloader = DeepCrackDownloader(download_path)
    downloader.ensure_downloaded()
    shutdown_logging()
