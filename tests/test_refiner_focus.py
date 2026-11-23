"""Synthetic D4 recurrence, aligned cached drafts, and dispatch regressions."""
import copy
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import train_refiner as runner
from looproad.models.refiner import LoopRefiner
from looproad.models.d4_refiner import trajectory,transform
from looproad.data.augmentation import d4_transform
from looproad.data.d4_cache import D4CacheDataset,D4DraftCache
from scripts import train_refiner as runner
from scripts import wave3_smoke as smoke
from tests import test_wave_two as previous


class RefinerFocusTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(11)
        model=LoopRefiner(8,16,1,4,patch=4,image_size=8).eval()
        torch.nn.init.normal_(model.output_head[-1].weight,std=.03)
        return model,torch.randn(1,8,2,2),torch.randn(1,1,8,8)

    def test_all_d4_inverse_and_numpy_match(self):
        x=torch.arange(64).reshape(1,1,8,8)
        for code in range(8):
            self.assertTrue(torch.equal(transform(transform(x,code),code,True),x))
            np.testing.assert_array_equal(transform(x,code).numpy(),d4_transform(x.numpy(),code))

    def test_identity_matches_original_and_hook_counts(self):
        m,f,y=self.fixture()
        original={k:v.clone()for k,v,_,_ in runner.inference_steps(m,f,y,[1,2,4],torch.device('cpu'))}
        for mode in ['S5a','S5b']:
            for k,v,_,_,count in trajectory(m,f,y,[1,2,4],mode,(0,)):
                self.assertTrue(torch.equal(v,original[k]));self.assertEqual(count,int(k))

    def test_eight_views_one_core_call_per_step(self):
        m,f,y=self.fixture();batches=[]
        hook=m.core.register_forward_hook(lambda mod,args,out:batches.append(out.shape[0]))
        for mode in ['S5a','S5b']:
            batches.clear();rows=list(trajectory(m,f.repeat(8,1,1,1),y,[1,2,4],mode))
            self.assertEqual(batches,[8]*4);self.assertEqual([r[-1]for r in rows],[1,2,4])
        hook.remove()

    def test_independent_views_equal_independent_original_loops(self):
        m,f,y=self.fixture();features=torch.randn(8,8,2,2)
        outputs=[]
        for c in range(8):
            last=list(runner.inference_steps(m,features[c:c+1],transform(y,c),[4],torch.device('cpu')))[-1][1]
            outputs.append(transform(last,c,True))
        expected=torch.stack(outputs).mean(0)
        actual=list(trajectory(m,features,y,[4],'S5b'))[-1][1]
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

    def test_shared_answer_keeps_independent_view_states(self):
        m,_,y=self.fixture();features=torch.randn(8,8,2,2)
        states=[];tokens=[]
        for c in range(8):
            state,token=m.init(features[c:c+1],transform(y,c))
            states.append(state);tokens.append(token)
        shared=y.clone()
        with torch.no_grad():
            for _ in range(4):
                deltas=[]
                for c in range(8):
                    before=transform(shared,c)
                    states[c],after=m.step(states[c],before,tokens[c])
                    deltas.append(transform(after-before,c,True))
                shared=shared+torch.stack(deltas).mean(0)
        actual=list(trajectory(m,features,y,[4],'S5a'))[-1][1]
        torch.testing.assert_close(actual,shared,rtol=1e-6,atol=1e-6)

    def test_streamed_official_scores_on_two_synthetic_images(self):
        from PIL import Image
        from scripts import evaluate_refiner_focus as evaluation
        model,features,draft=self.fixture()
        encoder=torch.nn.Sequential(torch.nn.AvgPool2d(4),torch.nn.Conv2d(3,8,1)).eval()
        head=torch.nn.Sequential(torch.nn.Conv2d(8,1,1),torch.nn.Upsample(size=(8,8))).eval()
        class Pool:
            def __init__(self,*a,initializer,initargs):initializer(*initargs)
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def apply_async(self,fn,args):return types.SimpleNamespace(get=lambda:fn(*args))
        context=types.SimpleNamespace(Pool=Pool)
        cache=types.SimpleNamespace(load=lambda ids,device:draft)
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            label=Path(directory)/'label.png';Image.fromarray(np.full((8,8),3,dtype=np.uint8)).save(label)
            rows=[(domain,domain,Path('synthetic'),label)for domain in runner.DOMAINS]
            with patch.object(evaluation.mp,'get_context',return_value=context), \
                 patch.object(evaluation,'read_image',return_value=torch.zeros(1,3,8,8)),redirect_stdout(io.StringIO()):
                report=evaluation.evaluate(encoder,head,model,rows,[1,2,4],torch.device('cpu'),['E0','S5a','S5b'],cache,regression=True)
            self.assertEqual(len(report['images']),2)
            for groups in report['groups'].values():
                self.assertEqual(groups['Overall']['4']['counts']['valid'],128)
            self.assertTrue(all(image['identity_regression_max_difference']==0 for image in report['images']))

    def test_augmented_dataset_and_cache_use_same_code(self):
        dataset=previous.Tiles('Train','Urban');wrapped=D4CacheDataset(dataset,7)
        source=dataset[0];sample=wrapped[0];code=int(sample[3].split('|d4=')[1])
        np.testing.assert_array_equal(sample[0],d4_transform(source[0],code))
        np.testing.assert_array_equal(sample[4],d4_transform(source[4],code))
        fake=types.SimpleNamespace(load=lambda ids,device:torch.from_numpy(source[1][None,None]))
        draft=D4DraftCache(fake).load([sample[3]],torch.device('cpu'))
        np.testing.assert_array_equal(draft.numpy()[0,0],sample[1])

    def test_default_loss_weights_rng_against_main(self):
        def historical(path):
            source=subprocess.check_output(['git','show','1dd5e1a:'+path.relative_to(runner.ROOT).as_posix()],cwd=runner.ROOT,text=True)
            module=types.ModuleType('prior_refiner');module.__file__=str(path)
            exec(compile(source,str(path),'exec'),module.__dict__);return module
        with patch.object(previous,'historical',side_effect=historical):
            previous.WaveTwoTests().test_refiner_default_main_matches_historical_loss_weights_and_rng()
        self.assertFalse(runner.build_parser().parse_args([]).d4_aug)

    def test_coordinator_import_and_dispatch_without_torch(self):
        program="""
import builtins,sys
from unittest.mock import patch
import torch
assert not torch.cuda.is_initialized()
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith('torch'): raise AssertionError(name)
    return original(name,*args,**kwargs)
with patch.object(builtins,'__import__',side_effect=guard):
    from scripts import wave3_smoke
    cmds=wave3_smoke.commands(wave3_smoke.ROOT/'runs/test-wave3')
    assert list(cmds)==['sg1','S5a','S5b','dg1','RS-D4','RW-D4','stress']
with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA init during import')):
    from scripts import evaluate_refiner_focus,train_refiner_focus
assert not torch.cuda.is_initialized()
"""
        r=subprocess.run([sys.executable,'-c',program],cwd=runner.ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_failure_does_not_stop_subsequent_stage(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as d:
            with patch.object(smoke.subprocess,'run',return_value=types.SimpleNamespace(returncode=1))as run,redirect_stdout(io.StringIO())as output:
                passed=smoke.run_stages({'a':['false'],'b':['false']},Path(d))
            self.assertEqual(run.call_count,2);self.assertFalse(passed)
            self.assertNotIn('WAVE3_SMOKE_COMPLETE',output.getvalue())
