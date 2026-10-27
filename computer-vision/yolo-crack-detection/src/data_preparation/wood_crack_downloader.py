"""Wood surface defects dataset downloader from Kaggle."""

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

WOOD_KAGGLE_ID = "nomihsa965/large-scale-image-dataset-of-wood-surface-defects"


class WoodCrackDownloader(DataDownloader):
    """Download wood surface defects dataset from Kaggle."""

    def __init__(self, download_path: Path):
        self._download_path = download_path

    @property
    def dataset_name(self) -> str:
        return "wood_crack"

    @property
    def download_path(self) -> Path:
        return self._download_path

    def download(self) -> Path:
        """Download wood defects dataset from Kaggle.

        Returns:
            Path to the downloaded dataset.
        """
        kaggle_path = kagglehub.dataset_download(WOOD_KAGGLE_ID)
        kaggle_path = Path(kaggle_path)
        self._download_path.mkdir(parents=True, exist_ok=True)
        self._organize_wood(kaggle_path)
        logger.info(f"Wood defects dataset downloaded to {self._download_path}")
        return self._download_path

    def _organize_wood(self, kaggle_path: Path):
        """Organize Kaggle wood dataset and print structure."""
        import shutil

        # Print what we got
        logger.info(f"Kaggle dataset structure:")
        for item in kaggle_path.rglob("*"):
            if item.is_dir():
                rel = item.relative_to(kaggle_path)
                logger.info(f"  [DIR]  {rel}")
            elif item.is_file():
                rel = item.relative_to(kaggle_path)
                logger.info(f"  [FILE] {rel}")

        # Copy everything
        for item in kaggle_path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(kaggle_path)
                dest = self._download_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(item, dest)

        logger.info(f"Total files copied: {len(list(self._download_path.rglob('*')))}")

    def _has_files(self) -> bool:
        """Check if wood dataset has files."""
        return self._download_path.exists() and any(
            self._download_path.rglob("*")
        )


if __name__ == '__main__':
    from src.utils.logging_setup import shutdown_logging

    download_path = Path(__file__).resolve().parents[2] / 'data' / 'external' / 'wood_crack'
    downloader = WoodCrackDownloader(download_path)
    downloader.ensure_downloaded()
    shutdown_logging()
