"""Pillow-based reader for the raw RoadSeg 0..7 labels."""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def map_raw_labels(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("RoadSeg masks must be single-channel integer arrays")
    if (raw < 0).any() or (raw > 7).any():
        raise ValueError("RoadSeg labels must be in 0..7")
    valid = raw != 0
    target = (raw == 3).astype(np.float32)
    return target, valid


class RoadDataset(Dataset):
    def __init__(self, split, domain, limit=None, seed=0):
        if split not in ("Train", "Val") or domain not in ("Urban", "Rural"):
            raise ValueError("split must be Train/Val and domain must be Urban/Rural")
        root_value = os.environ.get("EC523_DATA_DIR")
        if not root_value:
            raise EnvironmentError("Set EC523_DATA_DIR to the read-only RoadSeg dataset")
        root = Path(root_value) / split / domain
        image_dir, mask_dir = root / "images_png", root / "masks_png"
        pairs = [(path, mask_dir / path.name) for path in sorted(image_dir.glob("*.png"))]
        missing = [str(mask) for _, mask in pairs if not mask.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing mask for {missing[0]}")
        if limit is not None and len(pairs) > limit:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(pairs), size=limit, replace=False))
            pairs = [pairs[i] for i in indices]
        self.pairs = pairs
        self.domain = domain
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file)
        with Image.open(mask_path) as mask_file:
            raw = np.asarray(mask_file)
        if image.shape != (1024, 1024, 3) or image.dtype != np.uint8:
            raise ValueError(f"Expected 1024x1024 RGB uint8 image: {image_path}")
        if raw.shape != (1024, 1024) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"Expected 1024x1024 single-channel integer mask: {mask_path}")
        target, valid = map_raw_labels(raw)
        raw = np.array(raw, dtype=np.uint8, copy=True, order="C")
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return image, target, valid, image_path.stem, raw
