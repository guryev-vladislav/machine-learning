import argparse
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

from src.utils.logging_setup import get_logger, shutdown_logging

logger = get_logger(__name__)

try:
    from src.utils.config import Config
except ImportError as e:
    logger.error(f"Critical import error: {e}")
    sys.exit(1)

class ImagesToVideoConverter:
    def __init__(self, config=None):
        self.config = config or Config()
        self.fps = self.config.VIDEO_FPS
        self.width = self.config.VIDEO_WIDTH
        self.height = self.config.VIDEO_HEIGHT

        self.codec = cv2.VideoWriter_fourcc(*'MJPG')
        self.codec_name = 'MJPG'

    def create_video_from_deepcrack(self, output_path=None):
        if output_path is None:
            output_path = self.config.GENERATED_VIDEOS_PATH / "deepcrack_cracked.mp4"

        rgb_dir = self.config.DEEPCRACK_PATH / "rgb"
        logger.info(f"Creating video from DeepCrack (cracked): {rgb_dir}")

        return self._create_video_from_directory(rgb_dir, output_path, label='crack')

    def create_video_from_sdnet_cracked(self, output_path=None):
        if output_path is None:
            output_path = self.config.GENERATED_VIDEOS_PATH / "sdnet_cracked.mp4"

        logger.info(f"Creating video from SDNET (cracked)")

        all_images = []
        for category in ["Decks", "Pavements", "Walls"]:
            cracked_dir = self.config.SDNET_PATH / category / "Cracked"
            if cracked_dir.exists():
                images = self._get_images_from_dir(cracked_dir)
                all_images.extend(images)
                logger.info(f"  Found {len(images)} images in {category}/Cracked")

        return self._create_video_from_list(all_images, output_path, label='crack')

    def create_video_from_sdnet_non_cracked(self, output_path=None):
        if output_path is None:
            output_path = self.config.GENERATED_VIDEOS_PATH / "sdnet_non_cracked.mp4"

        logger.info(f"Creating video from SDNET (non-cracked)")

        all_images = []
        for category in ["Decks", "Pavements", "Walls"]:
            non_cracked_dir = self.config.SDNET_PATH / category / "Non-cracked"
            if non_cracked_dir.exists():
                images = self._get_images_from_dir(non_cracked_dir)
                all_images.extend(images)
                logger.info(f"  Found {len(images)} images in {category}/Non-cracked")

        return self._create_video_from_list(all_images, output_path, label='no_crack')

    def _get_images_from_dir(self, directory):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        images = []
        for ext in exts:
            images.extend(sorted(Path(directory).glob(ext)))
        return sorted(images)

    def _create_video_from_directory(self, directory, output_path, label=None):
        images = self._get_images_from_dir(directory)
        return self._create_video_from_list(images, output_path, label)

    def _create_video_from_list(self, image_paths, output_path, label=None):
        if not image_paths:
            logger.warning(f"No images found for {output_path}")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating video with {len(image_paths)} frames to {output_path}")
        logger.info(f"Video parameters: {self.width}x{self.height} @ {self.fps}fps, codec: {self.codec_name}")

        writer = cv2.VideoWriter(
            str(output_path),
            self.codec,
            self.fps,
            (self.width, self.height)
        )

        if not writer.isOpened():
            logger.error(f"Failed to initialize VideoWriter. Codec: {self.codec_name}, Size: {self.width}x{self.height}")
            return None

        successful_frames = 0
        for idx, img_path in enumerate(tqdm(image_paths, desc="Writing frames", leave=True, disable=False)):
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    logger.warning(f"Failed to read: {img_path}")
                    continue

                frame = cv2.resize(frame, (self.width, self.height))

                if label:
                    cv2.putText(
                        frame,
                        f"Label: {label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if label == 'crack' else (0, 0, 255),
                        2
                    )

                ret = writer.write(frame)
                if ret:
                    successful_frames += 1
                else:
                    logger.warning(f"Failed to write frame {idx}: {img_path}")
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}")
                continue

        writer.release()
        logger.info(f"Video saved to {output_path} ({successful_frames}/{len(image_paths)} frames written)")
        return output_path

    def create_all_videos(self):
        logger.info("Creating all dataset videos...")

        videos = {
            'deepcrack_cracked': self.create_video_from_deepcrack(),
            'sdnet_cracked': self.create_video_from_sdnet_cracked(),
            'sdnet_non_cracked': self.create_video_from_sdnet_non_cracked(),
        }

        logger.info("All videos created successfully!")
        for name, path in videos.items():
            if path:
                logger.info(f"  - {name}: {path}")

        return videos

def main():
    parser = argparse.ArgumentParser(description='Convert images to video')
    parser.add_argument('--source', type=str, default='all',
                        choices=['deepcrack', 'sdnet_cracked', 'sdnet_non_cracked', 'all'])
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--fps', type=int, default=None, help='FPS for output video')

    args = parser.parse_args()

    config = Config()
    if args.fps:
        config.VIDEO_FPS = args.fps

    config.create_directories()

    converter = ImagesToVideoConverter(config)

    if args.source == 'all':
        converter.create_all_videos()
    elif args.source == 'deepcrack':
        converter.create_video_from_deepcrack(args.output)
    elif args.source == 'sdnet_cracked':
        converter.create_video_from_sdnet_cracked(args.output)
    elif args.source == 'sdnet_non_cracked':
        converter.create_video_from_sdnet_non_cracked(args.output)

if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_logging()







