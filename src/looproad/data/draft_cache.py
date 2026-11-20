"""Identity-checked, batch-local TTA draft logit cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CacheIds(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, valid, identity, raw = self.dataset[index]
        identity = f"{self.dataset.split}/{self.dataset.domain}/{identity}"
        return image, target, valid, identity, raw


def write_draft(root, identity, logits, dtype):
    path = Path(root) / (identity + ".npz")
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Cache identity escapes cache root")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, image_id=np.array(identity), logits=np.asarray(logits, dtype=dtype))
    return path.relative_to(root).as_posix()


class DraftCache:
    def __init__(self, root, checkpoint=None, split=None, pending=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root/("pending.json" if pending else "manifest.json")).read_text())
        if checkpoint is not None and self.manifest["head_sha256"] != sha256(checkpoint):
            raise ValueError("Cached draft checkpoint identity differs")
        if split is not None and self.manifest["split_sha256"] != sha256(split):
            raise ValueError("Cached draft split identity differs")
        if not pending and not self.manifest.get("gates_passed"):
            raise ValueError("Cached draft gates have not passed")

    def load(self, identities, device):
        values = []
        for identity in identities:
            path = self.root/self.manifest["entries"][identity]
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Cache entry escapes cache root")
            with np.load(path, allow_pickle=False) as saved:
                if saved["image_id"].item() != identity:
                    raise ValueError("Cached draft image identity differs")
                value = saved["logits"]
                if value.dtype != np.dtype(self.manifest["dtype"]) or value.ndim != 3 or value.shape[0] != 1:
                    raise ValueError("Cached draft shape/dtype differs")
                if not np.isfinite(value).all():
                    raise ValueError("Cached draft is nonfinite")
                values.append(value.astype(np.float32))
        return torch.from_numpy(np.stack(values)).to(device)


def frozen_cached_draft(encoder, head, images, identities, cache):
    encoder.eval()
    head.eval()
    with torch.no_grad():
        features = encoder(images)
        if hasattr(head, "refiner_features"):
            features = head.refiner_features(features)
        draft = cache.load(identities, images.device)
    if tuple(draft.shape) != (images.shape[0], 1, *images.shape[-2:]):
        raise ValueError("Cached draft does not match image dimensions")
    return features, draft
