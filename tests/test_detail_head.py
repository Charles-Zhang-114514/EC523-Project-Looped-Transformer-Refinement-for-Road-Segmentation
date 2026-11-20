"""Synthetic detail initialization, conditioning, strict keys and legacy parity."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from scripts import train_refiner as runner
from looproad.models.refiner import LoopRefiner,load_initial_refiner
from looproad.models.detail_features import DetailFeatures
from looproad.models.d4_refiner import trajectory
from scripts import train_detail_refiner as driver
from scripts import wave4_smoke as smoke
from tests import test_wave_two as previous


class DetailHeadTests(unittest.TestCase):
    def models(self):
        old=LoopRefiner(8,16,1,4,patch=16,image_size=32,head_type='conv_upsample')
        detail=LoopRefiner(8,16,1,4,patch=16,image_size=32,head_type='conv_upsample',detail_head=True)
        load_initial_refiner(detail,old.state_dict())
        return old,detail

    def test_zero_gate_shapes_and_identical_outputs(self):
        old,detail=self.models();torch.nn.init.normal_(old.output_head.final.weight,std=.02)
        load_initial_refiner(detail,old.state_dict())
        f=torch.randn(2,8,2,2);rgb=torch.randn(2,3,32,32);y=torch.randn(2,1,32,32)
        shapes=[];gates=[]
        handles=[detail.output_head.detail_encoder.register_forward_hook(lambda m,a,o:shapes.append(tuple(o.shape))),
                 detail.output_head.detail_gate.register_forward_hook(lambda m,a,o:gates.append(o.detach()))]
        with torch.no_grad():
            expected=runner.refine_steps(old,f,y,4)
            actual=runner.refine_steps(detail,DetailFeatures(f,rgb),y,4)
        self.assertTrue(all(torch.equal(a,b)for a,b in zip(expected,actual)))
        self.assertEqual(shapes,[(2,32,8,8)]*4)
        self.assertTrue(all(torch.count_nonzero(g)==0 for g in gates))
        for h in handles:h.remove()

    def test_parameter_counts_and_unchanged_core(self):
        model=LoopRefiner(head_type='conv_upsample',detail_head=True)
        c=model.parameter_counts();self.assertEqual(c['detail_added'],21680)
        self.assertEqual(c['core'],789760);self.assertEqual(c['trainable_total'],1214705)
        self.assertLessEqual(c['trainable_total'],10_000_000)
        self.assertEqual(sum(p.numel()for p in model.output_head.detail_encoder.parameters()),14336)
        self.assertEqual(sum(p.numel()for p in model.output_head.detail_fusion.parameters()),7072)
        self.assertEqual(sum(p.numel()for p in model.output_head.detail_gate.parameters()),272)

    def test_detail_depends_on_state_rgb_and_current_answer(self):
        _,m=self.models();h=m.output_head
        torch.nn.init.normal_(h.final.weight,std=.1)
        torch.nn.init.normal_(h.detail_gate.weight,std=.1)
        tokens=torch.randn(1,4,16,requires_grad=True);rgb=torch.randn(1,3,32,32,requires_grad=True);answer=torch.randn(1,1,32,32,requires_grad=True)
        h(tokens,2,rgb,answer).square().mean().backward()
        for x in (tokens,rgb,answer):
            self.assertIsNotNone(x.grad);self.assertGreater(x.grad.abs().sum().item(),0)

    def test_initial_load_strict_legacy_keys(self):
        old,new=self.models();state=old.state_dict();bad=dict(state);bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):load_initial_refiner(new,bad)
        with self.assertRaises(ValueError):load_initial_refiner(new,dict(state,unexpected=torch.zeros(1)))
        self.assertTrue(all(torch.equal(v,new.state_dict()[k])for k,v in state.items()))

    def test_detail_core_count_and_bundle_methods(self):
        _,model=self.models();f=DetailFeatures(torch.randn(1,8,2,2),torch.randn(1,3,32,32));y=torch.randn(1,1,32,32)
        calls=[row[-1]for row in trajectory(model,f.float().clone(),y,[1,2,4],codes=(0,))]
        self.assertEqual(calls,[1,2,4])

    def test_disabled_model_loss_weights_rng_against_prior_source(self):
        path=runner.ROOT/'src/looproad/models/refiner.py'
        source=subprocess.check_output(['git','show','e238133:src/looproad/models/refiner.py'],cwd=runner.ROOT,text=True)
        old=types.ModuleType('old_refiner');exec(compile(source,str(path),'exec'),old.__dict__)
        f=torch.randn(1,8,2,2);y=torch.randn(1,1,32,32);target=torch.rand_like(y);results=[]
        for cls in [old.LoopRefiner,LoopRefiner]:
            torch.manual_seed(14);m=cls(8,16,1,4,patch=16,image_size=32,head_type='conv_upsample');opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01);losses=[]
            for _ in range(2):
                opt.zero_grad();out=runner.refine_steps(m,f,y,2);loss=sum(torch.nn.functional.binary_cross_entropy_with_logits(v,target)for v in out)/2
                loss.backward();opt.step();losses.append(loss.item())
            results.append((losses,copy.deepcopy(m.state_dict()),torch.get_rng_state()))
        self.assertEqual(results[0][0],results[1][0])
        self.assertTrue(all(torch.equal(v,results[1][1][k])for k,v in results[0][1].items()))
        self.assertTrue(torch.equal(results[0][2],results[1][2]))

    def test_default_training_main_config_and_rng_against_e238133(self):
        def historical(path):
            source=subprocess.check_output(['git','show','e238133:'+path.relative_to(runner.ROOT).as_posix()],cwd=runner.ROOT,text=True)
            m=types.ModuleType('prior_training');m.__file__=str(path);exec(compile(source,str(path),'exec'),m.__dict__);return m
        with patch.object(previous,'historical',side_effect=historical):
            previous.WaveTwoTests().test_refiner_default_main_matches_historical_loss_weights_and_rng()
        self.assertFalse(runner.build_parser().parse_args([]).detail_head)

    def test_formal_detail_training_through_official_evaluation(self):
        import io
        from contextlib import redirect_stdout
        class Pool:
            def __init__(self,*args,initializer,initargs):initializer(*initargs)
            def apply_async(self,fn,args):return types.SimpleNamespace(get=lambda:fn(*args))
            def close(self):pass
            def join(self):pass
            def terminate(self):pass
        encoder=torch.nn.Sequential(torch.nn.AvgPool2d(16),torch.nn.Conv2d(3,8,1)).requires_grad_(False)
        encoder.out_channels=8
        head=torch.nn.Sequential(torch.nn.Conv2d(8,1,1),torch.nn.Upsample(size=(32,32))).requires_grad_(False)
        cache=types.SimpleNamespace(load=lambda ids,device:torch.zeros(len(ids),1,32,32,device=device))
        datasets={d:previous.Tiles('Val',d)for d in runner.DOMAINS}
        original_model=runner.LoopRefiner;original_loader=runner.make_loader
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            out=Path(directory)
            argv=['train_refiner.py','--detail-head','--d4-aug','--draft-cache','runs/synthetic',
                  '--head-type','conv_upsample','--epochs','1','--batch-size','2','--dimension','16','--heads','4',
                  '--max-steps','2','--device','cpu','--loader-workers','0','--eval-loader-workers','0',
                  '--metric-workers','1','--limit-per-domain','1','--select-per-domain','1','--select-every','1',
                  '--val-split','configs/val_split_seed0.json','--eval-k','1,2,4',
                  '--out-dir',str(out/'report'),'--ckpt-dir',str(out/'checkpoints')]
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()), \
                 patch.object(runner.y0,'load_frozen_probe',return_value=(encoder,head,{'head':'conv_up'})), \
                 patch.object(runner,'LoopRefiner',side_effect=lambda *a,**kw:original_model(*a,image_size=32,**kw)), \
                 patch.object(runner,'DraftCache',return_value=cache), \
                 patch.object(runner.probe,'RoadDataset',side_effect=previous.Tiles), \
                 patch.object(runner,'protocol_datasets',return_value=datasets), \
                 patch.object(runner,'make_loader',side_effect=lambda d,b,w,*a,**kw:original_loader(d,b,0,*a,**kw)), \
                 patch.object(runner.mp,'get_context',return_value=types.SimpleNamespace(Pool=Pool)):
                runner.main()
            report=json.loads((out/'report/metrics.json').read_text())
            self.assertTrue(report['config']['detail_head'])
            self.assertEqual(len(report['epochs'][0]['batches']),2)
            self.assertEqual(len(report['evaluation']['images']),2)
            self.assertTrue((out/'checkpoints/best.pt').is_file())
            self.assertIn('update_seconds',report['epochs'][0]['batches'][0])

    def test_time_estimate_requires_successful_gpu_smoke(self):
        names=['DRG1','base-RS','detail-RS','score-RS','base-RW','detail-RW','score-RW']
        data={n:dict(passed=True)for n in names}
        for name in ['RS','RW']:
            data['base-'+name]['batches']=[dict(update_seconds=2.)]*20
            data['detail-'+name]['batches']=[dict(update_seconds=2.1)]*20
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            p=Path(directory);(p/'metrics.json').write_text(json.dumps(data))
            b=driver.estimate(p,'S');self.assertAlmostEqual(b['added_seconds_per_update'],.1)
            self.assertEqual(b['train_updates'],6310)
            self.assertGreater(driver.estimate(p,'W')['full_DRG1_allowance_seconds'],0)
            self.assertEqual(driver.estimate(p,'W')['inference_steps'],33388)
            data['DRG1']['passed']=False;(p/'metrics.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):driver.estimate(p,'S')

    def test_full_gate_failure_writes_reason(self):
        from scripts import detail_gate
        from scripts import evaluate_refiner_focus as focus
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            out=Path(directory)/'failed-gate'
            with patch.object(focus.runner.y0,'load_frozen_probe',return_value=(None,None,None)), \
                 patch.object(focus,'DraftCache',return_value=None), \
                 patch.object(detail_gate,'load_detail',return_value=None), \
                 patch.object(focus,'evaluate',return_value={'groups':{'E0':{}},'images':[None]*835}), \
                 patch.object(focus,'compare_groups',side_effect=AssertionError('maximum difference exceeds tolerance')):
                with self.assertRaises(SystemExit):detail_gate.run(out,full=True)
            gate=json.loads((out/'metrics.json').read_text())['DRG1']
            self.assertFalse(gate['passed']);self.assertIn('exceeds tolerance',gate['reason'])

    def test_failed_gate_prevents_formal_training(self):
        from scripts import detail_gate
        argv=['train_detail_refiner.py','--arm','W','--seed','0','--smoke-dir','runs/mock-smoke']
        with patch.object(sys,'argv',argv),patch.object(driver,'estimate',return_value={}), \
             patch.object(detail_gate,'run',side_effect=SystemExit('DRG1_FAILED')), \
             patch.object(runner,'main')as train:
            with self.assertRaises(SystemExit):driver.main()
            train.assert_not_called()

    def test_coordinator_import_dispatch_never_initializes_cuda(self):
        program="""
import builtins
from unittest.mock import patch
import torch
assert not torch.cuda.is_initialized()
old=builtins.__import__
def guard(name,*a,**kw):
    if name.startswith('torch'):raise AssertionError(name)
    return old(name,*a,**kw)
with patch.object(builtins,'__import__',side_effect=guard):
    from scripts import wave4_smoke,train_detail_refiner
    assert len(wave4_smoke.commands(wave4_smoke.ROOT/'runs/synthetic-wave4'))==7
with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA during import')):
    from scripts import detail_gate
assert not torch.cuda.is_initialized()
"""
        r=subprocess.run([sys.executable,'-c',program],cwd=runner.ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)
