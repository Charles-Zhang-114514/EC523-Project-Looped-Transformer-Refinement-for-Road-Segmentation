"""CPU-only CUDA ownership checks for dispatch and spawned CPU workers."""
import ast
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class CpuRows:
    def __len__(self):
        return 1

    def __getitem__(self,index):
        import numpy as np
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError("Data worker owns a CUDA context")
        return (np.zeros((3,32,32),dtype=np.float32),np.zeros((32,32),dtype=np.float32),
                np.ones((32,32),dtype=bool),"cuda_initialized=False",np.ones((32,32),dtype=np.uint8))


def audit_spawn_worker():
    import torch
    import numpy as np
    from unittest.mock import patch
    before=torch.cuda.is_initialized()
    with patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CPU worker attempted CUDA initialization")), \
         patch.object(torch.cuda,"init",side_effect=AssertionError("CPU worker attempted CUDA init")):
        for name in ("scripts.cache_tta_draft","scripts.train_refiner","scripts.train_lora_draft"):
            importlib.import_module(name)
        from scripts import train_refiner as runner
        imported=torch.cuda.is_initialized()
        runner.initialize_eval_worker(os.environ["EC523_STARTER_DIR"])
        runner.probe.initialize_data_worker(0)
        mask=np.zeros((32,32),dtype=np.uint8)
        labels=np.ones((32,32),dtype=np.uint8)
        score=runner.score_image("Urban","synthetic",{"y0_draft":mask,"1":mask},labels,False)
        return dict(before=before,after_import=imported,after_score=torch.cuda.is_initialized(),
                    valid=score[2]["y0_draft"]["counts"]["valid"])


class CudaDispatchTests(unittest.TestCase):
    def test_import_and_five_stage_dispatch_never_initialize_cuda(self):
        # A fresh interpreter also catches imports hidden by other unit tests.
        program = r'''
import builtins
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import torch

assert not torch.cuda.is_initialized()
original_import=builtins.__import__
def guarded_import(name,globals=None,locals=None,fromlist=(),level=0):
    if name.startswith("torch") or name=="scripts.cache_tta_draft" or "cache_tta_draft" in (fromlist or ()):
        raise AssertionError("Supervisor imported a CUDA-capable dependency")
    return original_import(name,globals,locals,fromlist,level)
with patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("Supervisor initialized CUDA")), \
     patch.object(builtins,"__import__",side_effect=guarded_import):
    from scripts import wave2_smoke as smoke
    called=[]
    def fake_process(command,**kwargs):
        if command[2]=="scripts/cache_tta_draft.py":
            arm="cache"
            root=Path(command[command.index("--cache-dir")+1])
            root.mkdir(parents=True)
            (root/"manifest.json").write_text(json.dumps(dict(gates_passed=True,smoke=True,total_images=12)))
        else:
            directory=Path(command[command.index("--out-dir")+1])
            arm=directory.parent.name if directory.name=="report" else directory.name
            directory.mkdir(parents=True)
            images=[dict(domain=d,image_id=f"{d}{i}") for d in ("Urban","Rural") for i in range(2)]
            data=dict(epochs=[dict(loss=.5,batches=[dict(loss=.5)]*20)],
                      evaluation=dict(images=images),frozen_head_unchanged=True)
            (directory/"metrics.json").write_text(json.dumps(data))
        called.append(arm)
        assert not torch.cuda.is_initialized()
        return subprocess.CompletedProcess(command,0)
    with tempfile.TemporaryDirectory(dir=smoke.ROOT/"runs") as directory:
        out=Path(directory)/"output"
        with patch.object(sys,"argv",["wave2_smoke.py","--out-dir",str(out)]), \
             patch.object(smoke.subprocess,"run",side_effect=fake_process):
            smoke.main()
        assert called==["cache","RS","RW","LA","LC"]
        report=json.loads((out/"metrics.json").read_text())
        assert all(report[arm]["exit_code"]==0 for arm in called)
assert not torch.cuda.is_initialized()
print("CPU_ONLY_DISPATCH_PASS")
'''
        completed=subprocess.run([sys.executable,"-c",program],cwd=ROOT,capture_output=True,text=True,timeout=90)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertIn("CPU_ONLY_DISPATCH_PASS",completed.stdout)
        self.assertIn("WAVE2_SMOKE_COMPLETE",completed.stdout)

    def test_spawn_imports_and_actual_metric_worker_do_not_initialize_cuda(self):
        with mp.get_context("spawn").Pool(1) as pool:
            result=pool.apply(audit_spawn_worker)
        self.assertFalse(result["before"])
        self.assertFalse(result["after_import"])
        self.assertFalse(result["after_score"])
        self.assertEqual(result["valid"],1024)

    def test_spawn_data_loader_initializer_and_numpy_collation_stay_on_cpu(self):
        import torch
        from scripts import train_linear_probe as probe
        loader=torch.utils.data.DataLoader(CpuRows(),batch_size=1,num_workers=1,
                    multiprocessing_context="spawn",worker_init_fn=probe.initialize_data_worker,
                    collate_fn=probe.collate_numpy_batch,pin_memory=False)
        batches=list(loader)
        self.assertEqual(list(batches[0])[3],["cuda_initialized=False"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_cache_default_builder_body_and_cli_dispatch_match_f799215(self):
        from scripts import cache_tta_draft as current
        path=Path(current.__file__)
        old_source=subprocess.check_output(["git","show","f799215:scripts/cache_tta_draft.py"],cwd=ROOT,text=True)
        old=types.ModuleType("prior_cache_entry")
        old.__file__=str(path)
        exec(compile(old_source,str(path),"exec"),old.__dict__)
        def body(source):
            node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=="build_cache")
            return ast.dump(node,include_attributes=False)
        self.assertEqual(body(old_source),body(path.read_text()))
        for module in (old,current):
            with patch.object(sys,"argv",["cache_tta_draft.py"]),patch.object(module,"build_cache") as build:
                module.main()
            self.assertEqual(build.call_args.args,(ROOT/"runs/cache/tta_draft_a1i_s0",))
            self.assertEqual(build.call_args.kwargs,{})
