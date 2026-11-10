"""Confirmation scheduling, frozen provenance and paired statistics; no real jobs."""
import contextlib
import copy
import io
import os
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import repeat_refiner_batch as repeat
from scripts import six_run_batch as base


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name).resolve()
        self.reference = self.repo/'runs/six_run/reference'
        self.env = {'EC523_DATA_DIR': '/fake/data', 'EC523_STARTER_DIR': '/fake/starter', 'HF_HOME': '/fake/cache'}
        self.runtime = {'gpu': 'NVIDIA L40S', 'torch': 'test', 'cuda': 'test', 'python': sys.version.split()[0]}
        self.torch = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='test'),
                                     cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: 'NVIDIA L40S'))
        environment = patch.dict(os.environ, {**self.env, 'JOB_ID': '12345'})
        environment.start()
        self.addCleanup(environment.stop)
        files = {}
        for relative in ('scripts/activate_scc.sh', 'scripts/train_refiner.py', str(base.PROBE)):
            original = self.reference/'source'/relative
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_text('seed-0 original\n')
            files[relative] = base.digest(original)
            working = self.repo/relative
            working.parent.mkdir(parents=True, exist_ok=True)
            working.write_text('seed-0 original\n')
        for relative in repeat.ADDITIONS:
            path = self.repo/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('new confirmation helper\n')
        base.write_json(self.reference/'manifest.json', {'schema': 1, 'seed': 0, 'job_id': 'old123',
                         'variants': list(base.VARIANTS), 'files': files, 'gpu_type': 'L40S',
                         'expected_environment': self.env})
        for method in repeat.METHODS:
            run = self.reference/'source/runs'/method
            (run/'reports').mkdir(parents=True)
            recipe = next(v for v in base.VARIANTS if v['name'] == method)
            groups = {domain: {k: {'scores': {'A': .5}, 'structure': {'fp_fragments': 40},
                                    'extra': {'new_fragments': {'count': 3},
                                              'tracking': {'gap_repair_fraction': .1, 'new_break_fraction': .01}}}
                               for k in ('y0_draft', '1', '2', '4', '6', '8')}
                      for domain in ('Overall', 'Urban', 'Rural')}
            config = {k:v for k,v in recipe.items() if k != 'name'}
            config.update(seed=0, batch_size=4, epochs=10, resume=False, eval_only=False,
                          out_dir=f'runs/{method}/reports', ckpt_dir=f'runs/{method}/checkpoints',
                          selection_ks={'1': .25, '2': .25, '4': .5})
            report = {'config': config, 'protocol': 'val_sel_rep', 'epochs': [{}]*10,
                      'evaluated_epoch': 4, 'selection_files': ['a'], 'evaluation_files': ['b'],
                      'evaluation': {'groups': groups}}
            base.write_json(run/'reports/metrics.json', report)
            base.write_json(run/'runtime.json', self.runtime)
            base.write_json(run/'command.json', ['/old/python', '-u', 'scripts/train_refiner.py', *base.training_arguments(recipe, 0)])
            (run/'SUCCESS').write_text('completed seed 0\n')
        (self.repo/'runs/six_run/LATEST').write_text('original pointer\n')

    def prepare(self):
        return repeat.prepare_batch(self.repo, self.reference)

    def complete(self, batch, item, a4=.5):
        run = batch/'source/runs'/item['name']
        (run/'reports').mkdir(parents=True)
        report = repeat.read_json(batch/'reference'/item['experiment']/'reports/metrics.json')
        report['config'].update(seed=item['seed'], out_dir=f'runs/{item["name"]}/reports', ckpt_dir=f'runs/{item["name"]}/checkpoints')
        report['evaluation']['groups']['Overall']['4']['scores']['A'] = a4
        base.write_json(run/'reports/metrics.json', report)
        base.write_json(run/'runtime.json', self.runtime)
        (run/'SUCCESS').write_text('complete')
        return run

    def test_exact_six_tasks_and_only_seed_output_arguments_change(self):
        expected = [('control',1), ('lovasz',1), ('augmentation',1), ('control',2), ('lovasz',2), ('augmentation',2)]
        self.assertEqual([(x['experiment'],x['seed']) for x in repeat.TASKS], expected)
        batch = self.prepare()
        for item in repeat.TASKS:
            command = repeat.training_command(batch, item)
            original = repeat.read_json(batch/'reference'/item['experiment']/'command.json')
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[command.index('--seed')+1], str(item['seed']))
            self.assertEqual(command[command.index('--out-dir')+1], f'runs/{item["name"]}/reports')
            self.assertEqual(command[command.index('--ckpt-dir')+1], f'runs/{item["name"]}/checkpoints')
            for flag in ('--seed','--out-dir','--ckpt-dir'):
                command[command.index(flag)+1] = original[original.index(flag)+1]
            self.assertEqual(command[1:], original[1:])
        self.assertFalse(list((batch/'source/runs').glob('*/checkpoints/best.pt')))

    def test_snapshot_uses_original_training_inputs_and_protects_seed0_evidence(self):
        (self.repo/'scripts/train_refiner.py').write_text('later working change')
        batch = self.prepare()
        self.assertEqual((batch/'source/scripts/train_refiner.py').read_text(), 'seed-0 original\n')
        manifest = repeat.read_manifest(batch)
        repeat.validate_batch(batch, manifest)
        (batch/'reference/control/reports/metrics.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'seed-0 evidence changed'):
            repeat.validate_batch(batch, manifest)
        self.assertEqual((self.repo/'runs/six_run/LATEST').read_text(), 'original pointer\n')

    def test_reference_configuration_mismatch_is_rejected(self):
        path = self.reference/'source/runs/lovasz/reports/metrics.json'
        report = repeat.read_json(path)
        report['config']['batch_size'] = 8
        base.write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'unexpected settings differ'):
            self.prepare()

    def test_report_rejects_wrong_seed_settings_membership_and_changed_draft(self):
        reference = repeat.read_json(self.reference/'source/runs/control/reports/metrics.json')
        for change, message in (
            (lambda r:r['config'].update(seed=2), 'wrong seed'),
            (lambda r:r['config'].update(batch_size=8), 'training settings differ'),
            (lambda r:r.update(evaluation_files=['different image']), 'evaluation_files differs'),
            (lambda r:r['evaluation']['groups']['Urban']['y0_draft']['scores'].update(A=.1), 'frozen-draft scores differ')):
            report = copy.deepcopy(reference)
            report['config']['seed'] = 1
            change(report)
            with self.assertRaisesRegex(ValueError, message):
                repeat.validate_report(report, reference, 1, 'test')

    def test_submission_is_one_array_and_does_not_move_screening_pointer(self):
        commands = []
        def output(command, **kwargs):
            commands.append(command)
            return 'L40s 20\n' if command[0] == 'qgpus' else '12345.1-6:1\n'
        with patch.object(repeat, 'ROOT', self.repo), patch.object(repeat, 'preflight'), \
                patch.object(repeat.subprocess, 'check_output', side_effect=output), \
                patch('sys.argv', ['repeat_refiner_batch.py', 'submit', '--reference', str(self.reference)]), \
                contextlib.redirect_stdout(io.StringIO()):
            repeat.main()
        batch = Path((self.repo/'runs/confirmation/LATEST').read_text().strip())
        self.assertEqual(repeat.read_manifest(batch)['job_id'], '12345')
        self.assertEqual(len([c for c in commands if c[0]=='qsub']), 1)
        command = commands[-1]
        self.assertIn('gpu_type=L40s', command)
        self.assertEqual(command[command.index('-tc')+1], '2')
        self.assertEqual(command[-2], str(batch/'source/jobs/refiner_repeat.qsub'))
        self.assertEqual((self.repo/'runs/six_run/LATEST').read_text(), 'original pointer\n')

    def test_preflight_failure_prevents_submission(self):
        with patch.object(repeat, 'ROOT', self.repo), patch.object(repeat, 'preflight', side_effect=RuntimeError('failed check')), \
                patch.object(repeat.subprocess, 'check_output', return_value='L40S 20\n') as execute, \
                patch('sys.argv', ['repeat_refiner_batch.py', 'submit', '--reference', str(self.reference)]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                repeat.main()
        self.assertEqual([call.args[0][0] for call in execute.call_args_list], ['qgpus'])
        self.assertFalse((self.repo/'runs/confirmation/LATEST').exists())

    def test_task_success_requires_valid_completed_report_and_refuses_overwrite(self):
        batch = self.prepare()
        item = repeat.TASKS[0]
        def train(command, **kwargs):
            run = self.complete(batch, item)
            (run/'SUCCESS').unlink()  # Only the task controller may write this marker.
        with patch.dict(sys.modules, {'torch': self.torch}), patch.object(repeat, 'preflight'), \
                patch.object(repeat.subprocess, 'run', side_effect=train), contextlib.redirect_stdout(io.StringIO()):
            repeat.task(batch, 1)
            self.assertTrue((batch/'source/runs/control_seed1/SUCCESS').is_file())
            with self.assertRaises(FileExistsError):
                repeat.task(batch, 1)

    def test_runtime_mismatch_fails_before_training(self):
        batch = self.prepare()
        self.torch.cuda.get_device_name = lambda i: 'NVIDIA A40'
        with patch.dict(sys.modules, {'torch': self.torch}), patch.object(repeat, 'preflight') as checks, \
                patch.object(repeat.subprocess, 'run') as train:
            with self.assertRaisesRegex(RuntimeError, 'GPU/software must match'):
                repeat.task(batch, 2)
            checks.assert_not_called()
            train.assert_not_called()
        self.assertTrue((batch/'source/runs/lovasz_seed1/FAILED.json').is_file())
        self.assertFalse((batch/'source/runs/lovasz_seed1/SUCCESS').exists())

    def test_sample_standard_deviation_and_matching_seed_differences(self):
        rows=[]
        for method, values in (('control', [.5,.6,.7]), ('lovasz', [.51,.62,.69])):
            for seed, value in enumerate(values):
                rows.append({'experiment': method, 'seed': seed, 'status':'complete', **{k:value for k in repeat.METRICS}})
        summary, paired = repeat.summarize(rows)
        self.assertAlmostEqual(summary['control']['metrics']['A4']['mean'], .6)
        self.assertAlmostEqual(summary['control']['metrics']['A4']['std'], .1)
        self.assertAlmostEqual(paired['lovasz']['A4']['mean'], .02/3)
        self.assertAlmostEqual(paired['lovasz']['A4']['std'], (.0007/3)**.5)
        for seed, value in enumerate((.01,.02,-.01)):
            self.assertAlmostEqual(paired['lovasz']['A4']['by_seed'][str(seed)], value)
        self.assertEqual(summary['augmentation']['status'], 'incomplete')
        self.assertNotIn('metrics', summary['augmentation'])

    def test_comparison_includes_seed0_and_requires_three_completed_seeds_for_means(self):
        batch = self.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            rows, summary, paired = repeat.compare(batch)
        self.assertEqual(sum(r['status']=='complete' for r in rows), 3)
        self.assertTrue(all(x['status']=='incomplete' and 'metrics' not in x for x in summary.values()))
        self.assertFalse(paired)
        for item in repeat.TASKS:
            self.complete(batch, item, .5+.01*item['seed'])
        with contextlib.redirect_stdout(io.StringIO()):
            rows, summary, paired = repeat.compare(batch)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(x['status']=='complete' for x in summary.values()))
        self.assertAlmostEqual(summary['control']['metrics']['A4']['mean'], .51)
        self.assertAlmostEqual(summary['control']['metrics']['A4']['std'], .01)
        self.assertEqual(paired['lovasz']['A4']['mean'], 0.)
        self.assertTrue((batch/'comparison.json').is_file())


if __name__ == '__main__':
    unittest.main()
