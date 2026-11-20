#!/usr/bin/env python3
"""GPU-only wave-2 smoke: cache gates, RS/RW updates, LA frozen BN and LC."""
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import cache_tta_draft as build
from scripts import train_lora_draft as lora_train
from scripts import train_refiner as runner
from looproad.data.augmentation import D4TrainingDataset
from looproad.data.draft_cache import CacheIds, frozen_cached_draft
from looproad.models.lora import LoRAEncoder


def main():
    if not torch.cuda.is_available():
        raise ValueError("Run on an allocated GPU")
    (ROOT/"runs/cache").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    report = {}
    with tempfile.TemporaryDirectory(prefix="wave2-smoke-", dir=ROOT/"runs/cache") as directory:
        encoder, head, baseline, cache, train_sets, val_sets = build.build_cache(Path(directory)/"draft", smoke=True)
        loader = runner.make_loader(ConcatDataset([CacheIds(ds) for ds in train_sets.values()]), 2, 0, True, 0)
        baseline_state = copy.deepcopy(baseline.state_dict())
        for arm in ("RS", "RW"):
            torch.manual_seed(0)
            model = runner.LoopRefiner(1024, head_type="conv_upsample").to(device)
            if arm == "RW":
                model.load_state_dict(baseline_state, strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if arm=="RS" else 1e-4, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: runner.lr_multiplier(step, 6310 if arm=="RS" else 1893, 500 if arm=="RS" else 100))
            iterator = iter(loader)
            rows = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(20):
                try:
                    images, target, valid, ids, _ = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    images, target, valid, ids, _ = next(iterator)
                images, target, valid = (runner.probe.as_device_tensor(x,device) for x in (images,target,valid))
                optimizer.zero_grad(set_to_none=True)
                start = runner.stamp(device)
                features, draft = frozen_cached_draft(encoder, head, images, ids, cache)
                outputs = runner.refine_steps(model, features, draft, 4)
                loss = runner.deep_supervision(outputs, target, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite smoke loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                scheduler.step()
                rows.append(dict(step=step,loss=float(loss.detach()),seconds=runner.stamp(device)-start,
                                 peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            scored = runner.evaluate(encoder,head,model,val_sets,[1,2,4,8],device,1,6,
                                      geometry=False,draft_cache=cache)
            report[arm] = dict(updates=rows,evaluation=scored)
            del model,optimizer,scheduler,outputs,loss
        del baseline,baseline_state,loader,iterator
        train_sets = [D4TrainingDataset(ds,i) for i,ds in enumerate(train_sets.values())]
        for arm in ("LA","LC"):
            torch.manual_seed(0)
            local_head = copy.deepcopy(head)
            local_head.requires_grad_(arm=="LC")
            before = {k:v.detach().cpu().clone() for k,v in local_head.state_dict().items()}
            local_encoder = LoRAEncoder(encoder.encoder).to(device) if arm=="LA" else encoder
            parameters = [p for p in local_encoder.parameters() if p.requires_grad]+[p for p in local_head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters,lr=2e-5 if arm=="LA" else 1e-4,weight_decay=0)
            loader = runner.make_loader(ConcatDataset(train_sets),2,0,True,0)
            torch.cuda.reset_peak_memory_stats()
            rows = lora_train.train_updates(local_encoder,local_head,loader,optimizer,20,1262 if arm=="LA" else 3155,
                      freeze_head=arm=="LA",lr=2e-5 if arm=="LA" else 1e-4,warmup=100 if arm=="LA" else 300)
            unchanged = all(torch.equal(value, local_head.state_dict()[key].detach().cpu()) for key,value in before.items())
            if arm=="LA" and (not unchanged or any(p.grad is not None for p in local_head.parameters())):
                raise ValueError("LA changed frozen head weights or BN statistics")
            scored = lora_train.draft.evaluate(local_encoder,local_head,val_sets,device,1,6)
            report[arm] = dict(updates=rows,head_and_BN_unchanged=unchanged,evaluation=scored)
            if arm=="LA":
                # Rebuild the frozen base for LC; its control must have no adapters.
                del local_encoder,optimizer,parameters
                encoder, head, _ = runner.y0.load_frozen_probe(build.HEAD,device)
        report["cache"] = cache.manifest
    out=ROOT/"runs/wave2/smoke"
    out.mkdir(parents=True,exist_ok=False)
    (out/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("WAVE2_SMOKE_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
