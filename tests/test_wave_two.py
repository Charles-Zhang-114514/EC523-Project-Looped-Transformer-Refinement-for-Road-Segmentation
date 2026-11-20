"""Wave-2 cached draft identities, frozen BN and legacy numerical regression."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset

from scripts import train_refiner as runner
from scripts import train_lora_draft as lora
from looproad.data.draft_cache import CacheIds, DraftCache, frozen_cached_draft, sha256, write_draft


class Tiles:
    def __init__(self, split, domain, **unused):
        self.split, self.domain = split, domain
        self.pairs = [(Path(domain+".png"), Path(domain+"-label.png"))]
    def __len__(self):
        return 1
    def __getitem__(self, index):
        image = np.random.default_rng(0).random((3,32,32),dtype=np.float32)
        raw = np.where(image[0]>.5,3,1).astype(np.uint8)
        return image,(raw==3).astype(np.float32),raw!=0,self.domain,raw


def historical(path):
    module = types.ModuleType("before_wave2_"+path.stem)
    module.__file__ = str(path)
    source = subprocess.check_output(["git","show","f799215:"+path.relative_to(runner.ROOT).as_posix()],
                                     cwd=runner.ROOT,text=True)
    exec(compile(source,module.__file__,"exec"),module.__dict__)
    return module


def models():
    encoder = nn.Sequential(nn.AvgPool2d(4),nn.Conv2d(3,8,1)).requires_grad_(False)
    encoder.out_channels = 8
    head = nn.Sequential(nn.Conv2d(8,1,1),nn.Upsample(scale_factor=4,mode="bilinear",align_corners=False))
    return encoder,head


class WaveTwoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_cache_roundtrip_and_identity_validation(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/"runs") as directory:
            root=Path(directory)
            head=root/"head";head.write_bytes(b"head")
            split=root/"split";split.write_bytes(b"split")
            value=np.random.default_rng(1).normal(size=(1,8,8)).astype(np.float32)
            name="Val/Urban/1"
            relative=write_draft(root,name,value,"float32")
            manifest=dict(entries={name:relative},dtype="float32",head_sha256=sha256(head),
                          split_sha256=sha256(split),gates_passed=True)
            (root/"manifest.json").write_text(json.dumps(manifest))
            cache=DraftCache(root,head,split)
            self.assertTrue(torch.equal(cache.load([name],torch.device("cpu")),torch.from_numpy(value[None])))
            with self.assertRaises(FileExistsError):
                write_draft(root,name,value,"float32")
            head.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                DraftCache(root,head,split)
            manifest["entries"][name]=write_draft(root,"Val/Urban/2",value,"float32")
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"image identity"):
                DraftCache(root).load([name],torch.device("cpu"))

    def test_freeze_head_preserves_parameters_BN_and_upstream_gradients(self):
        encoder=nn.Conv2d(3,8,1)
        head=nn.Sequential(nn.BatchNorm2d(8),nn.Conv2d(8,1,1)).requires_grad_(False)
        initial=copy.deepcopy(head.state_dict())
        lora.set_training(encoder,head,True)
        self.assertFalse(head[0].training)
        head(encoder(torch.randn(2,3,8,8))).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(p.grad is not None for p in encoder.parameters()))
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in initial.items()))

    def test_lora_default_updates_match_historical_losses_weights_and_rng(self):
        old=historical(Path(lora.__file__))
        encoder,head=models()
        fixture=[Tiles("Train","Urban")[0],Tiles("Train","Rural")[0]]
        results=[]
        for module in (old,lora):
            a,b=copy.deepcopy(encoder).requires_grad_(True),copy.deepcopy(head)
            loader=runner.make_loader(fixture,2,0)
            optimizer=torch.optim.AdamW(list(a.parameters())+list(b.parameters()),lr=1e-4,weight_decay=0)
            torch.manual_seed(7)
            with patch.object(runner.probe,"as_device_tensor",side_effect=lambda x,device:torch.as_tensor(x)), \
                 patch.object(runner,"stamp",return_value=0.), \
                 patch.object(torch.cuda,"max_memory_allocated",return_value=None),redirect_stdout(io.StringIO()):
                rows=module.train_updates(a,b,loader,optimizer,2,3155)
            results.append((rows,copy.deepcopy(a.state_dict()),copy.deepcopy(b.state_dict()),torch.get_rng_state()))
        self.assertEqual(results[0][0],results[1][0])
        for index in (1,2):
            self.assertTrue(all(torch.equal(v,results[1][index][k]) for k,v in results[0][index].items()))
        self.assertTrue(torch.equal(results[0][3],results[1][3]))

    def test_refiner_default_main_matches_historical_loss_weights_and_rng(self):
        old=historical(Path(runner.__file__))
        original_model=runner.LoopRefiner
        encoder,head=models()
        datasets={d:Tiles("Val",d) for d in runner.DOMAINS}
        groups={d:{k:{"scores":{"A":.5,"quality":.5}} for k in ("y0_draft","1","2","4","6","8")}
                for d in (*runner.DOMAINS,"Overall")}
        results=[]
        with tempfile.TemporaryDirectory(dir=runner.ROOT/"runs") as temporary:
            for index,module in enumerate((old,runner)):
                directory=Path(temporary)/str(index)
                argv=["train_refiner.py","--epochs","1","--batch-size","2","--device","cpu",
                      "--loader-workers","0","--eval-loader-workers","0","--metric-workers","1",
                      "--val-split","configs/val_split_seed0.json","--out-dir",str(directory/"report"),
                      "--ckpt-dir",str(directory/"checkpoints")]
                with patch.object(sys,"argv",argv),redirect_stdout(io.StringIO()), \
                     patch.object(module.y0,"load_frozen_probe",return_value=(copy.deepcopy(encoder),copy.deepcopy(head),{"head":"conv_up"})), \
                     patch.object(module,"LoopRefiner",side_effect=lambda *a,**kw:original_model(8,dimension=16,heads=4,patch=4,image_size=32)), \
                     patch.object(module.probe,"RoadDataset",side_effect=Tiles), \
                     patch.object(module,"protocol_datasets",return_value=datasets), \
                     patch.object(module,"evaluate",return_value={"groups":groups,"wall_seconds":0.}), \
                     patch.object(module,"summary_markdown",return_value="# Fixture"):
                    module.main()
                saved=torch.load(directory/"checkpoints/last.pt",map_location="cpu",weights_only=True)
                results.append((saved,torch.get_rng_state(),np.random.get_state(),random.getstate()))
        a,b=results
        self.assertEqual([v["loss"] for v in a[0]["epochs"]],[v["loss"] for v in b[0]["epochs"]])
        self.assertTrue(all(torch.equal(v,b[0]["state_dict"][k]) for k,v in a[0]["state_dict"].items()))
        self.assertTrue(torch.equal(a[1],b[1]))
        np.testing.assert_equal(a[2][1],b[2][1])
        self.assertEqual(a[2][2:],b[2][2:])
        self.assertEqual(a[3],b[3])
        ca={k:v for k,v in a[0]["config"].items() if k not in ("out_dir","ckpt_dir")}
        cb={k:v for k,v in b[0]["config"].items() if k not in ("out_dir","ckpt_dir")}
        self.assertEqual(ca,cb)

    def test_all_selection_and_cached_forward_skip_head(self):
        self.assertIsNone(runner.build_parser().parse_args(["--select-per-domain","all"]).select_per_domain)
        image=torch.randn(2,3,8,8)
        encoder=nn.Identity()
        head=nn.Conv2d(3,1,1)
        cache=types.SimpleNamespace(load=lambda ids,device:torch.ones(2,1,8,8))
        with patch.object(head,"forward",side_effect=AssertionError("Head must be skipped")):
            features,draft=frozen_cached_draft(encoder,head,image,["Train/Urban/1","Train/Rural/2"],cache)
        self.assertTrue(torch.equal(features,image))
        self.assertEqual(draft.shape,(2,1,8,8))

    def test_float16_cache_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/"runs") as directory:
            root=Path(directory)
            value=np.random.default_rng(3).normal(size=(1,8,8)).astype(np.float32)
            name="Train/Rural/7"
            entry=write_draft(root,name,value,"float16")
            (root/"manifest.json").write_text(json.dumps(dict(entries={name:entry},dtype="float16",gates_passed=True)))
            loaded=DraftCache(root).load([name],torch.device("cpu"))
            self.assertTrue(torch.equal(loaded,torch.from_numpy(value.astype(np.float16).astype(np.float32)[None])))
            self.assertEqual(CacheIds(Tiles("Train","Urban"))[0][3],"Train/Urban/Urban")

    def test_refiner_initialization_loads_weights_without_consuming_rng(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/"runs") as directory:
            path=Path(directory)/"init.pt"
            source=runner.LoopRefiner(8,dimension=16,heads=4,patch=4,image_size=32)
            target=copy.deepcopy(source)
            with torch.no_grad():
                next(source.parameters()).add_(.1)
            probe_path=runner.ROOT/"runs/draft_head/A1-i/best.pt"
            torch.save(dict(state_dict=source.state_dict(),config=dict(probe_ckpt=probe_path.relative_to(runner.ROOT).as_posix())),path)
            state=torch.get_rng_state().clone()
            runner.initialize_refiner(target,path,probe_path)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v,target.state_dict()[k]) for k,v in source.state_dict().items()))
            with self.assertRaisesRegex(ValueError,"different draft"):
                runner.initialize_refiner(target,path,runner.ROOT/"runs/other.pt")
