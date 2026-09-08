#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("no_crack", "crack")


def _create_image(class_name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    image = np.full((size, size, 3), 150, dtype=np.int16)
    image += rng.normal(0, 12, image.shape).astype(np.int16)
    image = np.clip(image, 0, 255).astype(np.uint8)

    if class_name == "crack":
        points = np.array(
            [
                [size // 5, size // 2],
                [size // 3, size // 2 - size // 8],
                [size // 2, size // 2 + size // 10],
                [size * 4 // 5, size // 3],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, (35, 35, 35), max(1, size // 32))

    return image


def create_dataset(output_path: Path, train_count: int, val_count: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for split, count in (("train", train_count), ("val", val_count)):
        for class_name in CLASS_NAMES:
            class_path = output_path / split / class_name
            class_path.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = _create_image(class_name, size, rng)
                image_path = class_path / f"{class_name}_{index:04d}.png"
                cv2.imwrite(str(image_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small synthetic YOLO classification dataset")
    parser.add_argument("--output", type=Path, default=Path("data/quickstart_dataset"))
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.train_count, args.val_count, args.size) < 1:
        parser.error("train-count, val-count and size must be positive")

    create_dataset(args.output, args.train_count, args.val_count, args.size, args.seed)
    print(f"Smoke dataset created at {args.output}")


if __name__ == "__main__":
    main()
