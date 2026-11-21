"""Train-only CPU draft corruption and deterministic, disjoint Train holdouts."""
import copy
import os
import time
import numpy as np
import torch
from .d4_cache import D4DraftCache


def corrupt_logits(logits, probability, rng):
    """Never inspect labels; thresholds use float32 sigmoid probabilities."""
    original=np.asarray(logits,dtype=np.float32)
    if original.ndim!=3 or original.shape[0]!=1:
        raise ValueError('Expected [1,H,W] logits')
    out=original.copy()
    stats=dict(selected=bool(rng.random()<probability),changed_pixels=0,gap_count=0,
               false_block_count=0,gap_sizes=[],ellipse_areas=[],ellipse_centers=[])
    if not stats['selected']:return out,stats
    from scipy.ndimage import distance_transform_edt
    from scipy.special import expit
    from skimage.morphology import skeletonize
    mask=expit(original[0])>=.5
    points=np.argwhere(skeletonize(np.ascontiguousarray(mask)))
    count=min(int(rng.integers(1,5)),len(points))
    height,width=mask.shape
    for index in rng.choice(len(points),size=count,replace=False) if count else []:
        y,x=map(int,points[index]);side=int(rng.integers(8,65))
        y0=max(0,y-side//2);y1=min(height,y+(side-side//2))
        x0=max(0,x-side//2);x1=min(width,x+(side-side//2))
        np.minimum(out[0,y0:y1,x0:x1],-4,out=out[0,y0:y1,x0:x1])
        stats['gap_sizes'].append(side)
    stats['gap_count']=count
    target=int(rng.integers(0,3))
    if target:
        distance=distance_transform_edt(~mask) if mask.any() else np.full(mask.shape,np.inf)
        centers=np.argwhere(distance>=32)
        for _ in range(target):
            for attempt in range(64):
                area=int(rng.integers(50,401));aspect=float(rng.uniform(.5,2.))
                rx=np.sqrt(area*aspect/np.pi);ry=np.sqrt(area/aspect/np.pi)
                mx,my=int(np.ceil(rx)),int(np.ceil(ry))
                allowed=centers[(centers[:,0]>=my)&(centers[:,0]<height-my)&
                                (centers[:,1]>=mx)&(centers[:,1]<width-mx)]
                if not len(allowed):continue
                y,x=map(int,allowed[int(rng.integers(len(allowed)))])
                yy,xx=np.ogrid[-my:my+1,-mx:mx+1];ellipse=(xx/rx)**2+(yy/ry)**2<=1
                pixels=int(ellipse.sum())
                if not 50<=pixels<=400:continue
                region=out[0,y-my:y+my+1,x-mx:x+mx+1]
                region[ellipse]=np.maximum(region[ellipse],4)
                stats['ellipse_areas'].append(pixels);stats['ellipse_centers'].append([y,x]);stats['false_block_count']+=1
                break
    stats['changed_pixels']=int(np.count_nonzero(out!=original))
    return out,stats


class CorruptDraftDataset:
    """Loader workers read/transform/corrupt y0; images and labels stay intact."""
    def __init__(self,dataset,cache,probability,seed,domain_index=0):
        source=dataset
        while not hasattr(source,'split')and hasattr(source,'dataset'):source=source.dataset
        if source.split!='Train':raise ValueError('Draft corruption is Train only')
        self.dataset,self.cache=dataset,cache
        self.probability,self.seed,self.domain_index=probability,seed,domain_index
        self.epoch=0

    def __len__(self):return len(self.dataset)

    def set_epoch(self,epoch):
        self.epoch=epoch
        if hasattr(self.dataset,'set_epoch'):self.dataset.set_epoch(epoch)

    def __getitem__(self,index):
        if torch.cuda.is_initialized():raise RuntimeError('Corruption must run in a CPU-only loader process')
        image,target,valid,identity,raw=self.dataset[index]
        start=time.perf_counter()
        cache=D4DraftCache(self.cache) if '|d4='in identity else self.cache
        draft=cache.load([identity],torch.device('cpu'))[0].numpy()
        loaded=time.perf_counter()
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,self.domain_index,index]))
        draft,stats=corrupt_logits(draft,self.probability,rng)
        stats.update(image_id=identity,corruption_seconds=time.perf_counter()-loaded,
                     cache_read_seconds=loaded-start,worker_pid=os.getpid())
        return image,target,valid,dict(image_id=identity,draft=draft,stats=stats),raw


class PreparedDrafts:
    """Move only the batch-local loader-produced logits to the model's device."""
    def load(self,records,device):
        return torch.from_numpy(np.stack([r['draft']for r in records])).to(device)


def split_holdout(datasets,count,limit=None):
    """Fixed seed 0 independently within each sorted domain; before augmentation."""
    if count<1:raise ValueError('Positive holdout size required')
    held={};used={};excluded={}
    for dataset in datasets:
        if dataset.split!='Train' or count>=len(dataset):raise ValueError('Holdout must leave training images')
        pairs=sorted(dataset.pairs,key=lambda p:p[0].name)
        chosen=set(map(int,np.random.default_rng(0).choice(len(pairs),count,replace=False)))
        heldset=copy.copy(dataset);heldset.pairs=[p for i,p in enumerate(pairs)if i in chosen]
        dataset.pairs=[p for i,p in enumerate(pairs)if i not in chosen]
        if limit is not None and len(dataset.pairs)>limit:
            indices=np.sort(np.random.default_rng(0).choice(len(dataset.pairs),limit,replace=False))
            dataset.pairs=[dataset.pairs[int(i)]for i in indices]
        prefix=f'Train/{dataset.domain}/'
        used[dataset.domain]=[prefix+p[0].stem for p in dataset.pairs]
        excluded[dataset.domain]=[prefix+p[0].stem for p in heldset.pairs]
        if set(used[dataset.domain])&set(excluded[dataset.domain]):raise AssertionError('Train/holdout overlap')
        held[dataset.domain]=heldset
    return held,dict(train_ids=used,holdout_ids=excluded,curve=[],best_epoch=None,best_score=None)
