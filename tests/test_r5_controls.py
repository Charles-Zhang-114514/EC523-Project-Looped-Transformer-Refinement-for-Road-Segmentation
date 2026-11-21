"""Synthetic CPU corruption, bounded corrections, holdout and legacy parity."""
import copy
from contextlib import redirect_stdout
import io
import json
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
from scripts import train_refiner_r5 as driver
from scripts import wave5_smoke as smoke
from looproad.data.training_controls import corrupt_logits,CorruptDraftDataset,PreparedDrafts,split_holdout
from looproad.data.d4_cache import D4CacheDataset
from looproad.models.refiner import LoopRefiner
from looproad.models.d4_refiner import trajectory
from tests import test_wave_two as previous


class TinyRoad:
    def __init__(self,split,domain,limit=None,seed=0):
        self.split,self.domain=split,domain
        self.pairs=[(Path(f'{i}.png'),Path(f'{i}-label.png'))for i in range(min(limit or 6,6))]
    def __len__(self):return len(self.pairs)
    def __getitem__(self,index):
        name=self.pairs[index][0].stem
        rng=np.random.default_rng(int(name));image=rng.random((3,32,32),dtype=np.float32)
        raw=np.where(image[0]>.5,3,1).astype(np.uint8)
        return image,(raw==3).astype(np.float32),raw!=0,name,raw


class TinyCache:
    def load(self,ids,device):
        draft=np.full((len(ids),1,32,32),-2,dtype=np.float32);draft[:,:,15:18,:]=2
        return torch.from_numpy(draft).to(device)


class InlinePool:
    def __init__(self,*a,initializer,initargs):initializer(*initargs)
    def apply_async(self,fn,args):return types.SimpleNamespace(get=lambda:fn(*args))
    def close(self):pass
    def join(self):pass
    def terminate(self):pass


class R5Tests(unittest.TestCase):
    def test_corruption_reproducible_and_geometrically_valid(self):
        from scipy.ndimage import distance_transform_edt
        logits=np.full((1,128,128),-2,dtype=np.float32);logits[:,:,63:66]=2
        original=logits.copy();ellipses=0
        for seed in range(8):
            a,stats=corrupt_logits(logits,1,np.random.default_rng(seed))
            b,other=corrupt_logits(logits,1,np.random.default_rng(seed))
            np.testing.assert_array_equal(a,b);self.assertEqual(stats,other)
            self.assertTrue(1<=stats['gap_count']<=4)
            self.assertTrue(all(8<=s<=64 for s in stats['gap_sizes']))
            self.assertTrue(all(50<=v<=400 for v in stats['ellipse_areas']))
            distance=distance_transform_edt(~(logits[0]>=0))
            self.assertTrue(all(distance[y,x]>=32 for y,x in stats['ellipse_centers']))
            self.assertTrue(0<=stats['false_block_count']<=2);ellipses+=stats['false_block_count']
            changed=a!=logits;self.assertTrue(np.all((a[changed]==-4)|(a[changed]==4)))
        self.assertGreater(ellipses,0);np.testing.assert_array_equal(logits,original)
        a,stats=corrupt_logits(logits,0,np.random.default_rng(0));np.testing.assert_array_equal(a,logits);self.assertFalse(stats['selected'])

    def test_loader_changes_draft_only_and_uses_local_rng(self):
        dataset=D4CacheDataset(TinyRoad('Train','Urban'),3)
        a=CorruptDraftDataset(dataset,TinyCache(),1,7,0)
        state=np.random.get_state();torch_state=torch.get_rng_state().clone()
        source=dataset[0];result=a[0]
        for k in [0,1,2,4]:np.testing.assert_array_equal(source[k],result[k])
        b=a[0];np.testing.assert_array_equal(result[3]['draft'],b[3]['draft'])
        np.testing.assert_equal(state,np.random.get_state());self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        batch=PreparedDrafts().load([result[3]],torch.device('cpu'));self.assertEqual(batch.shape,(1,1,32,32))
        with self.assertRaises(ValueError):CorruptDraftDataset(TinyRoad('Val','Urban'),TinyCache(),.5,0)

    def test_holdout_disjoint_fixed_and_order_independent(self):
        sets=[TinyRoad('Train',d)for d in runner.DOMAINS]
        held,info=split_holdout(sets,2)
        sets2=[TinyRoad('Train',d)for d in runner.DOMAINS]
        for d in sets2:d.pairs.reverse()
        _,info2=split_holdout(sets2,2)
        self.assertEqual(info,info2)
        for domain in runner.DOMAINS:
            self.assertEqual(len(held[domain]),2)
            self.assertFalse(set(info['train_ids'][domain])&set(info['holdout_ids'][domain]))
            self.assertEqual(len(info['train_ids'][domain]),4)

    def test_bound_zero_initialization_and_core_calls(self):
        m=LoopRefiner(8,16,1,4,patch=4,image_size=8,delta_bound=1.)
        f=torch.randn(1,8,2,2);y=torch.randn(1,1,8,8)
        out=list(trajectory(m,f,y,[1,2,4],codes=(0,)))
        self.assertTrue(all(torch.equal(row[1],y)for row in out))
        torch.nn.init.constant_(m.output_head[-1].bias,10)
        out=list(trajectory(m,f,y,[1,2,4],codes=(0,)))
        self.assertEqual([r[-1]for r in out],[1,2,4]);self.assertTrue(all(float(v)<=1 for v in m._delta_maxima))
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):LoopRefiner(delta_bound=value)

    def test_disabled_defaults_against_9bcd97d(self):
        def historical(path):
            source=subprocess.check_output(['git','show','9bcd97d:'+path.relative_to(runner.ROOT).as_posix()],cwd=runner.ROOT,text=True)
            m=types.ModuleType('before_r5');m.__file__=str(path);exec(compile(source,str(path),'exec'),m.__dict__);return m
        with patch.object(previous,'historical',side_effect=historical):previous.WaveTwoTests().test_refiner_default_main_matches_historical_loss_weights_and_rng()
        a=runner.build_parser().parse_args([]);self.assertEqual(a.draft_corrupt,0);self.assertEqual(a.train_holdout_per_domain,0);self.assertIsNone(a.delta_bound)
        path=runner.ROOT/'src/looproad/models/refiner.py';source=subprocess.check_output(['git','show','9bcd97d:src/looproad/models/refiner.py'],cwd=runner.ROOT,text=True)
        old=types.ModuleType('before_r5_model');old.__package__='looproad.models';exec(compile(source,str(path),'exec'),old.__dict__)
        f=torch.randn(1,8,2,2);y=torch.randn(1,1,8,8);results=[]
        for cls in [old.LoopRefiner,LoopRefiner]:
            torch.manual_seed(17);m=cls(8,16,1,4,patch=4,image_size=8);opt=torch.optim.AdamW(m.parameters(),lr=3e-4);losses=[]
            for _ in range(2):
                opt.zero_grad();loss=runner.refine_steps(m,f,y,4)[-1].square().mean();loss.backward();opt.step();losses.append(float(loss.detach()))
            results.append((losses,copy.deepcopy(m.state_dict()),torch.get_rng_state()))
        self.assertEqual(results[0][0],results[1][0]);self.assertTrue(torch.equal(results[0][2],results[1][2]))
        self.assertTrue(all(torch.equal(v,results[1][1][k])for k,v in results[0][1].items()))

    def test_three_formal_entries_on_synthetic_images(self):
        for arm in ['CD','BS','HO']:
            encoder=torch.nn.Sequential(torch.nn.AvgPool2d(16),torch.nn.Conv2d(3,8,1)).requires_grad_(False);encoder.out_channels=8
            head=torch.nn.Sequential(torch.nn.Conv2d(8,1,1),torch.nn.Upsample(size=(32,32))).requires_grad_(False)
            original_model=runner.LoopRefiner;original_loader=runner.make_loader
            datasets={d:TinyRoad('Val',d,limit=1)for d in runner.DOMAINS}
            with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
                out=Path(directory);cmd=driver.training_command(arm,0,out,smoke=True)
                def setarg(key,value):cmd[cmd.index(key)+1]=str(value)
                for k,v in [('--max-steps',2),('--device','cpu'),('--dimension',16),('--heads',4)]:
                    if k in cmd:setarg(k,v)
                    else:cmd.extend([k,str(v)])
                if arm=='HO':setarg('--train-holdout-per-domain',1)
                with patch.object(sys,'argv',cmd[2:]),redirect_stdout(io.StringIO()), \
                     patch.object(runner.y0,'load_frozen_probe',return_value=(encoder,head,{'head':'conv_up'})), \
                     patch.object(runner,'LoopRefiner',side_effect=lambda *a,**kw:original_model(*a,image_size=32,**kw)), \
                     patch.object(runner,'DraftCache',return_value=TinyCache()), \
                     patch.object(runner.probe,'RoadDataset',side_effect=TinyRoad), \
                     patch.object(runner,'protocol_datasets',return_value=datasets), \
                     patch.object(runner,'make_loader',side_effect=lambda d,b,w,*a,**kw:original_loader(d,b,0,*a,**{k:v for k,v in kw.items()if k!='spawn_workers'})), \
                     patch.object(runner.mp,'get_context',return_value=types.SimpleNamespace(Pool=InlinePool)):
                    runner.main()
                report=json.loads((out/'report/metrics.json').read_text());self.assertEqual(len(report['epochs'][0]['batches']),2)
                self.assertEqual(set(report['selection_curve'][0]['L_by_domain']),{'Overall','Urban','Rural'})
                if arm=='CD':self.assertFalse(report['evaluation_corruption_applied']);self.assertEqual(report['training_corruption']['samples'],8)
                if arm=='BS':self.assertTrue(all(v<=1 for b in report['epochs'][0]['batches']for v in b['delta_maxima']))
                if arm=='HO':
                    h=report['holdout'];self.assertEqual(len(h['curve']),1);self.assertIsNotNone(h['best_epoch'])
                    self.assertFalse(set(sum(h['train_ids'].values(),[]))&set(sum(h['holdout_ids'].values(),[])))

    def test_loader_spawn_executes_corruption_without_cuda(self):
        d=CorruptDraftDataset(D4CacheDataset(TinyRoad('Train','Urban',limit=1)),TinyCache(),1.,0)
        loader=runner.make_loader(d,1,1,spawn_workers=True)
        batches=list(loader);record=list(batches[0])[3][0]
        self.assertGreater(record['stats']['changed_pixels'],0);self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_loader_restores_bound_for_inference(self):
        from scripts.eval_stress import load_model
        model=LoopRefiner(1024,16,1,4,patch=16,head_type='conv_upsample',delta_bound=1.)
        config=dict(dimension=16,layers=1,heads=4,mlp_ratio=4.,patch=16,head_type='conv_upsample',
                    delta_bound=1.,val_split_sha256='fixture',reset_state=False,no_mask_feedback=False)
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            path=Path(directory)/'bounded.pt';torch.save(dict(state_dict=model.state_dict(),config=config,epoch=1),path)
            restored,_=load_model(path,torch.device('cpu'),'fixture')
            self.assertEqual(restored.delta_bound,1.)

    def test_eval_rejects_training_corruption(self):
        ds=CorruptDraftDataset(D4CacheDataset(TinyRoad('Train','Urban')),TinyCache(),1,0)
        with self.assertRaisesRegex(ValueError,'must never'):
            runner.evaluate(None,None,None,{'Urban':ds},[1],torch.device('cpu'))

    def test_failed_child_does_not_stop_following_stage(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            with patch.object(smoke.subprocess,'run',return_value=types.SimpleNamespace(returncode=1))as run,redirect_stdout(io.StringIO())as output:
                passed=smoke.run_stages({'CD':['false'],'BS':['false'],'HO':['false']},Path(directory))
            self.assertFalse(passed);self.assertEqual(run.call_count,3)
            self.assertNotIn('WAVE5_SMOKE_COMPLETE',output.getvalue())

    def test_budget_uses_measured_corruption_and_holdout_only(self):
        data={n:dict(passed=True)for n in ['CD','score-CD','BS','score-BS','HO','score-HO']}
        data['CD']['corruption']=[dict(corruption_seconds=.01)]*80
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs')as directory:
            root=Path(directory);(root/'metrics.json').write_text(json.dumps(data))
            self.assertAlmostEqual(driver.estimate(root,'CD')['added_CPU_seconds'],252.2)
            self.assertEqual(driver.estimate(root,'BS')['added_CPU_seconds'],0)
            self.assertAlmostEqual(driver.estimate(root,'HO')['added_holdout_seconds'],1176)
            data['HO']['passed']=False;(root/'metrics.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):driver.estimate(root,'CD')

    def test_supervisor_imports_and_commands_are_cuda_free(self):
        program="""
import builtins
from unittest.mock import patch
import torch
assert not torch.cuda.is_initialized()
original=builtins.__import__
def guard(name,*a,**kw):
    if name.startswith('torch'):raise AssertionError(name)
    return original(name,*a,**kw)
with patch.object(builtins,'__import__',side_effect=guard):
    from scripts import wave5_smoke
    commands=wave5_smoke.commands(wave5_smoke.ROOT/'runs/mock')
    assert len(commands)==6
assert not torch.cuda.is_initialized()
"""
        r=subprocess.run([sys.executable,'-c',program],cwd=runner.ROOT,capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
