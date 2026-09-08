#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='data/dataset')
parser.add_argument('--dst', default='data/dataset_quick')
parser.add_argument('--train-per-class', type=int, default=200)
parser.add_argument('--val-per-class', type=int, default=50)
args = parser.parse_args()

src = Path(args.src)
dst = Path(args.dst)

if not src.exists():
    raise SystemExit(f"Source dataset not found: {src}")

train_src = src / 'train'
val_src = src / 'val'
if not train_src.exists() or not val_src.exists():
    raise SystemExit(f"Expected train/ and val/ under {src}")

if dst.exists():
    print(f"Removing existing dst {dst}")
    shutil.rmtree(dst)

for split in ['train', 'val']:
    (dst / split).mkdir(parents=True, exist_ok=True)

for cls_dir in train_src.iterdir():
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    files = [p for p in cls_dir.glob('*') if p.is_file()]
    random.shuffle(files)
    take = min(len(files), args.train_per_class)
    dest_dir = dst / 'train' / cls_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in files[:take]:
        shutil.copy2(p, dest_dir / p.name)
    print(f"Train: class {cls_name} copied {take} / {len(files)}")

for cls_dir in val_src.iterdir():
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    files = [p for p in cls_dir.glob('*') if p.is_file()]
    random.shuffle(files)
    take = min(len(files), args.val_per_class)
    dest_dir = dst / 'val' / cls_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in files[:take]:
        shutil.copy2(p, dest_dir / p.name)
    print(f"Val: class {cls_name} copied {take} / {len(files)}")

print(f"Quick dataset created at: {dst}")

