import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from src.utils.config import Config
    from src.training.pipeline import TrainingPipeline
except ImportError as e:
    logger.error(f"Critical import error in {os.path.basename(__file__)}: {e}")
    sys.exit(1)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    pipeline = TrainingPipeline()

    sample_images = [
        "data/external/deepcrack/rgb/001.jpg",
        "data/external/deepcrack/rgb/011.jpg",
        "data/external/deepcrack/rgb/017.jpg",
        "data/external/deepcrack/rgb/018.jpg",
        "data/external/sdnet2018/Decks/Cracked/7001 - 2.jpg",
        "data/external/sdnet2018/Decks/Cracked/7001 - 17.jpg",
        "data/external/sdnet2018/Decks/Non-cracked/7001-1.jpg",
        "data/external/sdnet2018/Decks/Non-cracked/7001-3.jpg",
        "data/external/sdnet2018/Decks/Non-cracked/7001-4.jpg",
        "data/external/sdnet2018/Decks/Non-cracked/7001-5.jpg",
    ]

    pipeline.run_full_experiment(test_images_paths=sample_images)

if __name__ == "__main__":
    main()