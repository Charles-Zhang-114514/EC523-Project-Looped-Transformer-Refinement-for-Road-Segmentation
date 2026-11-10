"""Deterministic train-only rotations/reflections before the frozen encoder."""
import numpy as np
from torch.utils.data import Dataset


def d4_transform(array, code, spatial_axes=(-2, -1)):
    if code not in range(8):
        raise ValueError('D4 code must be in 0..7')
    output = np.rot90(array, k=code % 4, axes=spatial_axes)
    if code >= 4:
        output = np.flip(output, axis=spatial_axes[1])
    return np.ascontiguousarray(output)


class D4TrainingDataset(Dataset):
    def __init__(self, dataset, seed=0):
        if dataset.split != 'Train':
            raise ValueError('Augmentation is permitted only on Train')
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        image, target, valid, image_id, raw = self.dataset[index]
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        code = int(generator.integers(8))
        return (d4_transform(image, code), d4_transform(target, code),
                d4_transform(valid, code), image_id, d4_transform(raw, code))
