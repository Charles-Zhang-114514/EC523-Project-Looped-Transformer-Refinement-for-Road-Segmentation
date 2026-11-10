"""Snapshot, scheduler submission and comparison safety without submitting jobs."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import six_run_batch as batcher


class SixRunBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        for relative in (batcher.PROBE, Path('scripts/check_refiner_inputs.py'),
                         Path('scripts/activate_scc.sh'), Path('scripts/train_refiner.py'),
                         Path('src/untracked_experiment.py'), Path('jobs/refiner_six.qsub')):
            path = self.repo/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original test input\n')
        self.env = patch.dict(os.environ, {'EC523_DATA_DIR': '/fake/data', 'EC523_STARTER_DIR': '/fake/starter',
                                          'HF_HOME': '/fake/cache', 'JOB_ID': '12345'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def prepare(self):
        with patch.object(batcher.subprocess, 'check_output', return_value='git provenance\n'):
            return batcher.prepare_batch(self.repo)

    def report(self, variant):
        groups = {domain: {k: {'scores': {'A': .5}, 'structure': {'fp_fragments': 40},
                              'extra': {'new_fragments': {'count': 3},
                                        'tracking': {'gap_repair_fraction': .1, 'new_break_fraction': .01}}}
                           for k in ('1', '2', '4')} for domain in ('Urban', 'Rural', 'Overall')}
        return {'config': {**{k:v for k,v in variant.items() if k != 'name'}, 'seed': 0,
                           'selection_ks': {'1': .25, '2': .25, '4': .5},
                           'objective_state': {'effective_aux_weight': 2.}},
                'protocol': 'val_sel_rep', 'epochs': [{}]*10, 'evaluated_epoch': 3,
                'best_score': .51, 'selection_files': ['a'], 'evaluation_files': ['b'],
                'evaluation': {'groups': groups}}

    def completed_run(self, batch, variant):
        run = batch/'source/runs'/variant['name']
        (run/'reports').mkdir(parents=True)
        batcher.write_json(run/'reports/metrics.json', self.report(variant))
        batcher.write_json(run/'runtime.json', {'gpu': 'NVIDIA L40S', 'torch': 'test', 'cuda': 'test', 'python': 'test'})
        (run/'SUCCESS').write_text('completed')
        return run

    def test_scheduler_gpu_model_uses_inventory_spelling(self):
        self.assertEqual(batcher.gpu_name('L40S', 'GPU total\nL40s 48\nA40 24\n'), 'L40s')
        with self.assertRaises(ValueError):
            batcher.gpu_name('L40S', 'A40 24')
        with self.assertRaises(ValueError):
            batcher.gpu_name('L40S,other=1', 'L40S,other=1 24')

    def test_snapshot_includes_working_changes_and_detects_mutation(self):
        batch = self.prepare()
        manifest = batcher.read_manifest(batch)
        self.assertIn('src/untracked_experiment.py', manifest['files'])
        self.assertIn(str(batcher.PROBE), manifest['files'])
        (self.repo/'src/untracked_experiment.py').write_text('later edit')
        batcher.validate_snapshot(batch, manifest)
        (batch/'source'/batcher.PROBE).write_text('changed probe')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            batcher.validate_snapshot(batch, manifest)

    def test_six_recipes_change_only_the_named_intervention(self):
        from scripts.train_refiner import build_parser
        configurations = [vars(build_parser().parse_args(batcher.training_arguments(v, 0))) for v in batcher.VARIANTS]
        allowed = {'pos_weight', 'loss', 'augmentation', 'out_dir', 'ckpt_dir'}
        base = {k:v for k,v in configurations[0].items() if k not in allowed}
        for recipe, config in zip(batcher.VARIANTS, configurations):
            self.assertEqual({k:v for k,v in config.items() if k not in allowed}, base)
            self.assertEqual(config['epochs'], 10)
            self.assertEqual(config['head_type'], 'conv_upsample')
            self.assertEqual(config['false_add_warmup_steps'], 500)
            for key in ('pos_weight', 'loss', 'augmentation'):
                self.assertEqual(config[key], recipe[key])

    def test_submission_command_and_latest_pointer_without_real_qsub(self):
        commands = []

        def output(command, **kwargs):
            commands.append(command)
            if command[0] == 'qgpus': return 'L40s 20\n'
            if command[0] == 'qsub': return '12345.1-6:1\n'
            return 'git provenance\n'

        with patch.object(batcher, 'ROOT', self.repo), patch.object(batcher, 'run_tests'), \
                patch.object(batcher, 'check_inputs'), \
                patch.object(batcher.subprocess, 'check_output', side_effect=output), \
                patch('sys.argv', ['six_run_batch.py', 'submit']), contextlib.redirect_stdout(io.StringIO()):
            batcher.main()
        batch = Path((self.repo/'runs/six_run/LATEST').read_text().strip())
        self.assertEqual(batcher.read_manifest(batch)['job_id'], '12345')
        command = next(c for c in commands if c[0] == 'qsub')
        self.assertIn('gpu_type=L40s', command)
        self.assertEqual(command[command.index('-tc')+1], '2')
        self.assertEqual(command[-1], str(batch))

    def test_failed_preflight_never_submits(self):
        with patch.object(batcher, 'ROOT', self.repo), patch.object(batcher, 'prepare_batch', return_value=self.repo), \
                patch.object(batcher, 'run_tests', side_effect=RuntimeError('test failed')), \
                patch.object(batcher.subprocess, 'check_output', return_value='L40s 20\n') as execute, \
                patch('sys.argv', ['six_run_batch.py', 'submit']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                batcher.main()
        self.assertEqual([call.args[0][0] for call in execute.call_args_list], ['qgpus'])
        self.assertFalse((self.repo/'runs/six_run/LATEST').exists())

    def test_task_marks_success_only_after_training_and_protects_previous_attempt(self):
        batch = self.prepare()
        run = batch/'source/runs/control'

        def execute(command, **kwargs):
            if 'scripts/train_refiner.py' in command:
                (run/'reports').mkdir()
                batcher.write_json(run/'reports/metrics.json', self.report(batcher.VARIANTS[0]))

        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.get_device_name', return_value='NVIDIA L40S'), \
                patch.object(batcher, 'run_tests'), patch.object(batcher.subprocess, 'run', side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()):
            batcher.task(batch, 1)
            self.assertTrue((run/'SUCCESS').is_file())
            with self.assertRaises(FileExistsError):
                batcher.task(batch, 1)
        self.assertFalse((run/'FAILED.json').exists())

    def test_task_failure_is_visible_and_never_success(self):
        batch = self.prepare()
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.get_device_name', return_value='NVIDIA L40S'), \
                patch.object(batcher, 'run_tests'), \
                patch.object(batcher.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['training'])), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                batcher.task(batch, 2)
        run = batch/'source/runs/posw3'
        self.assertTrue((run/'FAILED.json').is_file())
        self.assertFalse((run/'SUCCESS').exists())

    def test_comparison_reports_missing_runs_and_rejects_confounded_results(self):
        batch = self.prepare()
        self.completed_run(batch, batcher.VARIANTS[0])
        run = self.completed_run(batch, batcher.VARIANTS[1])
        with contextlib.redirect_stdout(io.StringIO()):
            rows = batcher.compare(batch)
        self.assertEqual([r['status'] for r in rows], ['complete', 'complete'] + ['incomplete']*4)
        self.assertEqual(rows[1]['delta_A4'], 0.)
        self.assertTrue((batch/'comparison.txt').is_file())
        report_path = run/'reports/metrics.json'
        report = json.loads(report_path.read_text())
        report['evaluation_files'] = ['different validation images']
        batcher.write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, 'evaluation_files differs'):
            batcher.compare(batch)
        report['evaluation_files'] = ['b']
        report['config']['seed'] = 42
        batcher.write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, 'other experiment settings differ'):
            batcher.compare(batch)


if __name__ == '__main__':
    unittest.main()
