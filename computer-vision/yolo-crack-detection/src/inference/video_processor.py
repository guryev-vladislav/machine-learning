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

class VideoProcessor:
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model_path = model_path
        self.detector = None

    def _init_detector(self):
        if self.detector is None:
            try:
                from src.models.yolo_crack_detector import YOLOCrackDetector
                self.detector = YOLOCrackDetector(config=self.config, model_path=self.model_path)
            except ImportError as e:
                logger.error(f"Failed to load YOLO detector: {e}")
                raise

    def process_video(self, video_path, output_path=None, draw_results=True):
        self._init_detector()

        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None

        if output_path is None:
            output_path = self.config.OUTPUTS_PATH / f"{video_path.stem}_detected.mp4"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output will be saved to: {output_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video info: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        source_fps = fps if fps > 0 else 1.0
        out_fps = self.config.OUTPUT_VIDEO_FPS or source_fps
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            out_fps,
            (width, height)
        )

        if not writer.isOpened():
            logger.error(f"Failed to create video writer")
            cap.release()
            return None

        frame_count = 0
        crack_frames = []
        no_crack_frames = []

        pbar = tqdm(total=total_frames, desc="Processing frames", leave=False)

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                result = self.detector.classify_frame(frame)

                if result:
                    is_crack = result['is_crack']
                    confidence = result['confidence']

                    if is_crack:
                        crack_frames.append(frame_count)
                    else:
                        no_crack_frames.append(frame_count)

                    if draw_results:
                        frame = self._draw_result(
                            frame,
                            is_crack=is_crack,
                            confidence=confidence
                        )

                writer.write(frame)
                pbar.update(1)
        finally:
            pbar.close()
            cap.release()
            writer.release()

        logger.info(f"Video processing completed!")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Frames with cracks: {len(crack_frames)}")
        logger.info(f"  Frames without cracks: {len(no_crack_frames)}")
        crack_ratio = len(crack_frames) / frame_count * 100 if frame_count else 0.0
        logger.info(f"  Crack ratio: {crack_ratio:.1f}%")

        self._save_report(
            output_path,
            frame_count,
            crack_frames,
            no_crack_frames,
            fps
        )

        return output_path

    def _draw_result(self, frame, is_crack, confidence):
        h, w = frame.shape[:2]
        color = (0, 0, 255) if is_crack else (0, 255, 0)
        label = "CRACK DETECTED" if is_crack else "NO CRACK"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 80), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )

        return frame

    def _save_report(self, video_path, total_frames, crack_frames, no_crack_frames, fps):
        video_path = Path(video_path)
        report_path = video_path.with_name(f'{video_path.stem}_report.txt')
        safe_fps = fps if fps > 0 else 1.0

        with open(report_path, 'w') as f:
            f.write(f"Video Processing Report\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Duration: {total_frames / safe_fps:.2f} seconds\n\n")

            f.write(f"Detection Results:\n")
            f.write(f"  Crack frames: {len(crack_frames)}\n")
            f.write(f"  No-crack frames: {len(no_crack_frames)}\n")
            crack_ratio = len(crack_frames) / total_frames * 100 if total_frames else 0.0
            f.write(f"  Crack ratio: {crack_ratio:.1f}%\n\n")

            if crack_frames:
                f.write(f"Frames with cracks (first 100):\n")
                for frame_num in crack_frames[:100]:
                    f.write(f"  Frame {frame_num} (time: {frame_num / safe_fps:.2f}s)\n")
                if len(crack_frames) > 100:
                    f.write(f"  ... and {len(crack_frames) - 100} more\n")

        logger.info(f"Report saved to: {report_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process video and detect cracks')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--no-draw', action='store_true', help='Do not draw results on video')

    args = parser.parse_args()

    config = Config()
    processor = VideoProcessor(model_path=args.model, config=config)

    result_path = processor.process_video(
        args.video,
        output_path=args.output,
        draw_results=not args.no_draw
    )

    if result_path:
        logger.info(f"Processing complete. Result: {result_path}")

if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_logging()





