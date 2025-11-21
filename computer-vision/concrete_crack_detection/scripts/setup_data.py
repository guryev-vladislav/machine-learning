# scripts/setup_data.py
import logging
import shutil
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_path} to {extract_to}")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        raise


def setup_deepcrack(data_dir):
    deepcrack_dir = Path(data_dir) / "external" / "deepcrack"
    if not deepcrack_dir.exists():
        logger.info("DeepCrack dataset setup required")
        logger.info("Please download DeepCrack dataset and place in data/external/deepcrack/")
        return False
    return True


def setup_metu(data_dir):
    metu_dir = Path(data_dir) / "external" / "metu"
    if not metu_dir.exists():
        logger.info("METU dataset setup required")
        logger.info("Please download METU dataset and place in data/external/metu/")
        return False
    return True


def setup_sdnet(data_dir):
    sdnet_dir = Path(data_dir) / "external" / "sdnet2018"
    if not sdnet_dir.exists():
        logger.info("SDNET2018 dataset setup required")
        logger.info("Please download SDNET2018 dataset and place in data/external/sdnet2018/")
        return False
    return True


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up dataset structure")

    deepcrack_ready = setup_deepcrack(data_dir)
    metu_ready = setup_metu(data_dir)
    sdnet_ready = setup_sdnet(data_dir)

    if all([deepcrack_ready, metu_ready, sdnet_ready]):
        logger.info("All datasets are ready for training")
    else:
        logger.info("Some datasets require setup. Please check the instructions above.")


if __name__ == "__main__":
    main()