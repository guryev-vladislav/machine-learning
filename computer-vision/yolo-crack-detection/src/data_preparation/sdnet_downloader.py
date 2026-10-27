"""SDNET2018 dataset downloader using KaggleHub."""

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

SDNET_KAGGLE_ID = "aniruddhsharma/structural-defects-network-concrete-crack-images"


class SDNETDownloader(DataDownloader):
    """Download SDNET2018 structural defects dataset from Kaggle."""

    def __init__(self, download_path: Path):
        self._download_path = download_path

    @property
    def dataset_name(self) -> str:
        return "sdnet2018"

    @property
    def download_path(self) -> Path:
        return self._download_path

    def download(self) -> Path:
        """Download SDNET2018 from Kaggle and organize into expected structure.

        Expected Kaggle structure:
            SDNET2018/
                Decks/
                    Cracked/
                    Non-cracked/
                Pavements/
                    Cracked/
                    Non-cracked/
                Walls/
                    Cracked/
                    Non-cracked/

        Returns:
            Path to the organized dataset.
        """
        kaggle_path = kagglehub.dataset_download(SDNET_KAGGLE_ID)
        kaggle_path = Path(kaggle_path)
        self._download_path.mkdir(parents=True, exist_ok=True)
        self._organize_sdnet(kaggle_path)
        logger.info(f"SDNET2018 downloaded and organized to {self._download_path}")
        return self._download_path

    def _organize_sdnet(self, kaggle_path: Path):
        """Organize Kaggle SDNET structure into expected categories."""
        import shutil

        categories = ["Decks", "Pavements", "Walls"]

        for category in categories:
            for label in ["Cracked", "Non-cracked"]:
                # Try different possible folder names from Kaggle
                possible_dirs = [
                    kaggle_path / category / label,
                    kaggle_path / category / f"{label}s",
                    kaggle_path / f"{category}_{label}",
                    kaggle_path / f"{category}{label}",
                ]

                source_dir = None
                for d in possible_dirs:
                    if d.exists():
                        source_dir = d
                        break

                if source_dir is None:
                    logger.warning(
                        f"Could not find {category}/{label} in Kaggle dataset. "
                        f"Checked: {[str(p) for p in possible_dirs]}"
                    )
                    continue

                dest_dir = self._download_path / category / label
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy files, skip existing
                files = list(source_dir.glob("*"))
                copied = 0
                for src_file in files:
                    if src_file.is_file():
                        dest_file = dest_dir / src_file.name
                        if not dest_file.exists():
                            shutil.copy2(src_file, dest_file)
                            copied += 1

                logger.info(
                    f"  Copied {copied} files: {category}/{label} -> {dest_dir}"
                )

    def _has_files(self) -> bool:
        """Check if SDNET dataset has files in all expected categories."""
        categories = ["Decks", "Pavements", "Walls"]
        labels = ["Cracked", "Non-cracked"]

        for category in categories:
            for label in labels:
                dir_path = self._download_path / category / label
                if not dir_path.exists():
                    return False
                if not any(dir_path.iterdir()):
                    return False

        return True


if __name__ == '__main__':
    from src.utils.logging_setup import shutdown_logging

    download_path = Path(__file__).resolve().parents[2] / 'data' / 'external' / 'sdnet2018'
    downloader = SDNETDownloader(download_path)
    downloader.ensure_downloaded()
    shutdown_logging()
