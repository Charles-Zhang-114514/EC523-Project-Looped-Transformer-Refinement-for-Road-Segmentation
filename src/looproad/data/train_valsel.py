"""Explicit Train + Val-sel pools and predeclared checkpoint identities."""
from pathlib import Path


def dataset_ids(dataset):
    return [f"{dataset.split}/{dataset.domain}/{pair[0].stem}" for pair in dataset.pairs]


def add_valsel_training(train_sets, valsel_sets, valrep_sets):
    """Assert TG1 from metadata before constructing augmentation or CUDA."""
    if any(ds.split != "Train" for ds in train_sets):
        raise ValueError("The base training pool must be Train")
    additions = list(valsel_sets.values())
    if any(ds.split != "Val" for ds in additions):
        raise ValueError("The additional training pool must be Val-sel")
    def identities(datasets):
        result = [identity for ds in datasets for identity in dataset_ids(ds)]
        if len(set(result)) != len(result):
            raise ValueError("Duplicate image identity in a training/evaluation pool")
        return result
    train = identities(train_sets)
    valsel = identities(additions)
    valrep = identities(list(valrep_sets.values()))
    combined = train + valsel
    if len(set(combined)) != len(combined) or set(combined) & set(valrep):
        raise ValueError("TG1 failed: training overlaps Val-rep or has duplicates")
    pool = list(train_sets) + additions
    assert set(identities(pool)) == set(train) | set(valsel)
    return pool, dict(passed=True, train_ids=train, valsel_ids=valsel,
                      training_ids=combined, valrep_ids=valrep,
                      training_count=len(combined), train_count=len(train),
                      valsel_count=len(valsel), valrep_count=len(valrep),
                      valrep_intersection=0)


def fixed_epoch_path(directory, epoch):
    if epoch < 1:
        raise ValueError("Fixed epoch must be positive")
    return Path(directory) / f"epoch_{epoch:03d}.pt"
