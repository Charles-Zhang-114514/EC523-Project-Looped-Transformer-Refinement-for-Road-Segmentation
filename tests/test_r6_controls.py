"""Synthetic-only TVS protocol, fixed checkpoints and default parity."""
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
import torch
from scripts import train_refiner as runner
from scripts import train_refiner_r6 as driver
from scripts import wave6_smoke as smoke
from looproad.data.train_valsel import add_valsel_training, fixed_epoch_path
from looproad.data.d4_cache import D4CacheDataset
from looproad.data.training_controls import CorruptDraftDataset
from tests.test_r5_controls import TinyRoad, TinyCache, InlinePool
from tests import test_wave_two as previous


class Road(TinyRoad):
    def __init__(self, split, domain, limit=None, seed=0):
        super().__init__(split, domain, limit, seed)
        if split == 'Val':
            self.pairs = [(Path(f'{int(a.stem)+10}.png'), b) for a,b in self.pairs]


def protocol(split, part, limit=None):
    result = {d:Road('Val',d) for d in runner.DOMAINS}
    for ds in result.values():
        ds.pairs = ds.pairs[:2] if part == 'Val-sel' else ds.pairs[2:4]
        if limit is not None:ds.pairs = ds.pairs[:limit]
    return result


class R6Tests(unittest.TestCase):
    def test_disabled_defaults_match_68d1641(self):
        def historical(path):
            source = subprocess.check_output(['git','show','68d1641:'+path.relative_to(runner.ROOT).as_posix()], cwd=runner.ROOT, text=True)
            module = types.ModuleType('before_r6');module.__file__ = str(path)
            exec(compile(source,str(path),'exec'),module.__dict__)
            return module
        with patch.object(previous,'historical',side_effect=historical):
            previous.WaveTwoTests().test_refiner_default_main_matches_historical_loss_weights_and_rng()
        a = runner.build_parser().parse_args([])
        self.assertFalse(a.train_include_valsel);self.assertIsNone(a.fixed_epoch)

    def test_training_union_disjoint_and_leak_rejected(self):
        train = [Road('Train',d,limit=2) for d in runner.DOMAINS]
        sel,rep = protocol(None,'Val-sel'),protocol(None,'Val-rep')
        state = torch.get_rng_state().clone()
        pool,meta = add_valsel_training(train,sel,rep)
        self.assertEqual(len(pool),4);self.assertEqual(meta['training_count'],8)
        self.assertEqual(set(meta['training_ids']),set(meta['train_ids'])|set(meta['valsel_ids']))
        self.assertFalse(set(meta['training_ids'])&set(meta['valrep_ids']))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        sel['Urban'].pairs += rep['Urban'].pairs[:1]
        with self.assertRaisesRegex(ValueError,'TG1'):
            add_valsel_training(train,sel,rep)

    def test_valsel_requires_explicit_permission_and_retains_cache_ids(self):
        ds = protocol(None,'Val-sel')['Urban']
        with self.assertRaises(ValueError):D4CacheDataset(ds)
        aug = D4CacheDataset(ds,3,allow_valsel=True)
        self.assertTrue(aug[0][3].startswith('Val/Urban/10|d4='))
        with self.assertRaises(ValueError):CorruptDraftDataset(aug,TinyCache(),.5,0)
        corrupt = CorruptDraftDataset(aug,TinyCache(),.5,0,allow_valsel=True)
        item = corrupt[0]
        self.assertTrue(item[3]['image_id'].startswith('Val/Urban/10|d4='))
        for index in [0,1,2,4]:
            self.assertTrue((item[index] == aug[0][index]).all())

    def test_fixed_epoch_path(self):
        self.assertEqual(fixed_epoch_path(Path('runs/x'),2),Path('runs/x/epoch_002.pt'))
        with self.assertRaises(ValueError):fixed_epoch_path(Path('runs/x'),0)

    def test_tvs_main_uses_epoch_two_despite_training_epoch_three(self):
        encoder = torch.nn.Sequential(torch.nn.AvgPool2d(16),torch.nn.Conv2d(3,8,1)).requires_grad_(False)
        encoder.out_channels = 8
        head = torch.nn.Sequential(torch.nn.Conv2d(8,1,1),torch.nn.Upsample(size=(32,32))).requires_grad_(False)
        original_model,original_loader,original_eval = runner.LoopRefiner,runner.make_loader,runner.evaluate
        evaluated = []
        def evaluate(*args,**kwargs):
            if 'geometry' not in kwargs:evaluated.append(copy.deepcopy(args[2].state_dict()))
            return original_eval(*args,**kwargs)
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs') as directory:
            root = Path(directory);cmd = driver.training_command('TVS',0,root,smoke=True)
            def setting(k,v):cmd[cmd.index(k)+1] = str(v)
            for k,v in [('--epochs',3),('--max-steps',6),('--device','cpu')]:setting(k,v)
            cmd += ['--dimension','16','--heads','4']
            with patch.object(sys,'argv',cmd[2:]),redirect_stdout(io.StringIO()), \
                 patch.object(runner.y0,'load_frozen_probe',return_value=(encoder,head,{'head':'conv_up'})), \
                 patch.object(runner,'LoopRefiner',side_effect=lambda *a,**kw:original_model(*a,image_size=32,**kw)), \
                 patch.object(runner,'DraftCache',return_value=TinyCache()), \
                 patch.object(runner.probe,'RoadDataset',side_effect=Road), \
                 patch.object(runner,'protocol_datasets',side_effect=protocol), \
                 patch.object(runner,'make_loader',side_effect=lambda d,b,w,*a,**kw:original_loader(d,b,0,*a,**{k:v for k,v in kw.items() if k!='spawn_workers'})), \
                 patch.object(runner.mp,'get_context',return_value=types.SimpleNamespace(Pool=InlinePool)), \
                 patch.object(runner,'is_new_best',side_effect=AssertionError('TVS must never select')), \
                 patch.object(runner,'evaluate',side_effect=evaluate):
                runner.main()
            report = json.loads((root/'report/metrics.json').read_text())
            self.assertEqual(report['evaluated_epoch'],2);self.assertEqual(report['checkpoint_source'],'epoch_002.pt')
            self.assertEqual(report['selection_curve'],[]);self.assertEqual(report['selection_files'],{})
            self.assertEqual([len(e['batches']) for e in report['epochs']],[2,2,2])
            self.assertEqual(len(report['descriptive_curve']),3)
            self.assertTrue(all(p['descriptive_only'] and not p['evaluation_corruption_applied'] for p in report['descriptive_curve']))
            self.assertEqual(report['config']['scheduler_total_steps'],6)
            saved = [torch.load(fixed_epoch_path(root/'checkpoints',i),map_location='cpu',weights_only=True) for i in [1,2,3]]
            for i,payload in enumerate(saved,1):self.assertEqual(payload['epoch'],i)
            self.assertTrue(all(torch.equal(v,evaluated[-1][k]) for k,v in saved[1]['state_dict'].items()))
            self.assertTrue(any(not torch.equal(v,saved[2]['state_dict'][k]) for k,v in saved[1]['state_dict'].items()))
            self.assertFalse(report['evaluation_corruption_applied'])
            self.assertEqual(report['training_protocol']['valrep_intersection'],0)
            self.assertIn('descriptive', (root/'report/summary.md').read_text().lower())

    def test_formal_commands_match_preregistered_settings(self):
        for arm in ['P8','W','TVS']:
            for seed in range(3):
                cmd = driver.training_command(arm,seed,Path('runs/mock'))
                a = runner.build_parser().parse_args(cmd[3:])
                self.assertTrue(a.d4_aug);self.assertFalse(a.detail_head)
                self.assertEqual(a.seed,seed);self.assertEqual(a.draft_corrupt,.8 if arm=='P8' else .5)
                self.assertIsNone(a.select_per_domain)
                self.assertEqual(a.epochs,3 if arm=='W' else 10)
                self.assertEqual(a.lr,1e-4 if arm=='W' else 3e-4)
                self.assertEqual(a.warmup_steps,100 if arm=='W' else 500)
                self.assertEqual(a.train_include_valsel,arm=='TVS')
                self.assertEqual(a.fixed_epoch,2 if arm=='TVS' else None)
                self.assertEqual(a.init_refiner,Path('runs/exp/e6_convdraft_s0/best.pt') if arm=='W' else None)
                self.assertEqual(driver.checkpoint_for(arm,Path('runs/mock')).name,'epoch_002.pt' if arm=='TVS' else 'best.pt')

    def test_failed_children_are_independent(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs') as directory:
            cmds = smoke.commands(Path(directory))
            with patch.object(smoke.subprocess,'run',return_value=types.SimpleNamespace(returncode=1)) as run,redirect_stdout(io.StringIO()) as output:
                self.assertFalse(smoke.run_stages(cmds,Path(directory)))
            self.assertEqual(run.call_count,6)
            self.assertNotIn('WAVE6_SMOKE_COMPLETE',output.getvalue())

    def test_supervisor_import_and_schedule_do_not_import_torch(self):
        program = """
import builtins
from unittest.mock import patch
import torch
assert not torch.cuda.is_initialized()
original=builtins.__import__
def guard(name,*a,**kw):
    if name.startswith('torch'):raise AssertionError(name)
    return original(name,*a,**kw)
with patch.object(builtins,'__import__',side_effect=guard):
    from scripts import wave6_smoke
    assert len(wave6_smoke.commands(wave6_smoke.ROOT/'runs/mock'))==6
assert not torch.cuda.is_initialized()
"""
        result = subprocess.run([sys.executable,'-c',program],cwd=runner.ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_fixed_epoch_validation_fails_before_model_loading(self):
        commands = [
            ['--fixed-epoch','2'],
            ['--train-include-valsel','--fixed-epoch','2'],
            ['--train-include-valsel','--fixed-epoch','11','--epochs','10',
             '--val-split','configs/val_split_seed0.json','--draft-cache','runs/cache/tta_draft_a1i_s0'],
        ]
        for args in commands:
            with patch.object(sys,'argv',['train_refiner.py']+args),redirect_stdout(io.StringIO()), \
                 patch('sys.stderr',new=io.StringIO()), \
                 patch.object(runner.y0,'load_frozen_probe',side_effect=AssertionError('No model should load')):
                with self.assertRaises(SystemExit):runner.main()

    def test_formal_requires_all_smoke_stages_and_budget_is_bounded(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs') as directory:
            root = Path(directory)
            report = {key:dict(passed=True) for key in ['P8','score-P8','W','score-W','TVS','score-TVS']}
            (root/'metrics.json').write_text(json.dumps(report));driver.require_smoke(root)
            report['TVS']['passed'] = False
            (root/'metrics.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):driver.require_smoke(root)
            prior = root/'runs/wave5/smoke_8821468';prior.mkdir(parents=True)
            (prior/'metrics.json').write_text(json.dumps(dict(CD=dict(corruption=[dict(corruption_seconds=.058771465166501)]))))
            with patch.object(driver,'ROOT',root):
                self.assertEqual(driver.estimate('P8')['h_rt'],'01:44:00')
                self.assertEqual(driver.estimate('W')['h_rt'],'00:48:00')
                self.assertEqual(driver.estimate('TVS')['h_rt'],'03:37:00')
