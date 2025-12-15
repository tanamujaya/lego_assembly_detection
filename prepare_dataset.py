#!/usr/bin/env python3
"""
Prepare combined dataset - FIXED FOR PNG IMAGES
"""

import shutil
from pathlib import Path
import random


def prepare_combined():
    print("=" * 60)
    print("PREPARING COMBINED DATASET")
    print("=" * 60)

    # Source paths
    renders_img = Path('./data/renders/images')
    renders_lbl = Path('./data/renders/labels')
    real_img = Path('./data/real_photos/images')
    real_lbl = Path('./data/real_photos/labels')

    # Output path
    output = Path('./data/combined_dataset')

    # Create output structure
    for split in ['train', 'val']:
        (output / split / 'images').mkdir(parents=True, exist_ok=True)
        (output / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Get all renders (PNG!)
    render_images = list(renders_img.glob('*.png'))
    print(f"\n📦 Found {len(render_images)} render images (.png)")

    # Split renders 80/20
    random.seed(42)
    random.shuffle(render_images)
    split_idx = int(len(render_images) * 0.8)

    render_train = render_images[:split_idx]
    render_val = render_images[split_idx:]

    print(f"   Train: {len(render_train)}")
    print(f"   Val: {len(render_val)}")

    # Copy render train
    print("\nCopying render training images...")
    for img in render_train:
        shutil.copy2(img, output / 'train' / 'images' / img.name)
        lbl = renders_lbl / img.with_suffix('.txt').name
        if lbl.exists():
            shutil.copy2(lbl, output / 'train' / 'labels' / lbl.name)

    # Copy render val
    print("Copying render validation images...")
    for img in render_val:
        shutil.copy2(img, output / 'val' / 'images' / img.name)
        lbl = renders_lbl / img.with_suffix('.txt').name
        if lbl.exists():
            shutil.copy2(lbl, output / 'val' / 'labels' / lbl.name)

    # Get all real photos (PNG!)
    real_images = list(real_img.glob('*.png'))
    print(f"\n📸 Found {len(real_images)} real photo images (.png)")

    # Split real photos 80/20
    random.shuffle(real_images)
    split_idx = int(len(real_images) * 0.8)

    real_train = real_images[:split_idx]
    real_val = real_images[split_idx:]

    print(f"   Train: {len(real_train)}")
    print(f"   Val: {len(real_val)}")

    # Copy real train
    print("\nCopying real photo training images...")
    for img in real_train:
        shutil.copy2(img, output / 'train' / 'images' / f"real_{img.name}")
        lbl = real_lbl / img.with_suffix('.txt').name
        if lbl.exists():
            shutil.copy2(lbl, output / 'train' / 'labels' / f"real_{lbl.name}")

    # Copy real val
    print("Copying real photo validation images...")
    for img in real_val:
        shutil.copy2(img, output / 'val' / 'images' / f"real_{img.name}")
        lbl = real_lbl / img.with_suffix('.txt').name
        if lbl.exists():
            shutil.copy2(lbl, output / 'val' / 'labels' / f"real_{lbl.name}")

    # Create dataset.yaml
    yaml_path = output / 'dataset.yaml'
    yaml_content = f"""path: {str(output.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 2
names: [correct, incorrect]
"""
    yaml_path.write_text(yaml_content)

    # Summary
    total_train = len(render_train) + len(real_train)
    total_val = len(render_val) + len(real_val)

    print("\n" + "=" * 60)
    print("✅ DATASET PREPARED!")
    print("=" * 60)
    print(f"Location: {output}")
    print(f"\nTrain images: {total_train}")
    print(f"  - Renders: {len(render_train)}")
    print(f"  - Real: {len(real_train)}")
    print(f"\nVal images: {total_val}")
    print(f"  - Renders: {len(render_val)}")
    print(f"  - Real: {len(real_val)}")
    print(f"\nTotal: {total_train + total_val}")
    print(f"Dataset YAML: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_combined()