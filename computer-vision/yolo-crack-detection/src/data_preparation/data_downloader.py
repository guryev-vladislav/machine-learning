"""Base class for dataset downloaders."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class DataDownloader(ABC):
    """Abstract base class for dataset downloaders."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset (e.g., 'sdnet2018', 'deepcrack')."""
        pass

    @property
    @abstractmethod
    def download_path(self) -> Path:
        """Path where the dataset should be stored."""
        pass

    @abstractmethod
    def download(self) -> Path:
        """Download the dataset and return the path to extracted files."""
        pass

    def ensure_downloaded(self) -> Path:
        """Download dataset only if it doesn't exist yet.

        Returns:
            Path to the downloaded dataset.
        """
        if self.download_path.exists() and self._has_files():
            logger.info(f"Dataset '{self.dataset_name}' already exists at {self.download_path}")
            return self.download_path

        logger.info(f"Dataset '{self.dataset_name}' not found. Downloading...")
        return self.download()

    @abstractmethod
    def _has_files(self) -> bool:
        """Check if the download path contains any image files."""
        pass
