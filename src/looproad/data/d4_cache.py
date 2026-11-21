"""Train-only D4 augmentation with exactly aligned cached draft logits."""
import numpy as np
import torch
from .augmentation import d4_transform


class D4CacheDataset:
    def __init__(self, dataset, seed=0, *, allow_valsel=False):
        if dataset.split != 'Train' and not (allow_valsel and dataset.split == 'Val'):
            raise ValueError('Train only')
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        image, target, valid, identity, raw = self.dataset[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        code = int(rng.integers(8))
        identity = f'{self.dataset.split}/{self.dataset.domain}/{identity}|d4={code}'
        return (d4_transform(image,code), d4_transform(target,code),
                d4_transform(valid,code), identity, d4_transform(raw,code))


class D4DraftCache:
    def __init__(self, cache):
        self.cache = cache

    def load(self, identities, device):
        parsed = [identity.rsplit('|d4=',1) for identity in identities]
        drafts = self.cache.load([p[0] for p in parsed],torch.device('cpu'))
        values = [d4_transform(value.numpy(),int(code)) for value,(_,code) in zip(drafts,parsed)]
        return torch.from_numpy(np.stack(values)).to(device)
