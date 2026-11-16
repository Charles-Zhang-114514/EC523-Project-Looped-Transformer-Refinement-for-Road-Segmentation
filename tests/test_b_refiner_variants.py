"""B controls: legacy numerical parity, isolated K RNG and preserved reports."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_refiner as runner
from analysis import aggregate_experiments as aggregate
from tests.test_a2_conv_draft import fixture_report, APPENDIX

ROOT = runner.ROOT
JOB = ROOT / 'jobs/b_refiner_variants.qsub'


def historical_module(path):
    module = types.ModuleType('pre_B_' + path.stem)
    module.__file__ = str(path)
    source = subprocess.check_output(['git', 'show', '4e17ef3:' + path.relative_to(ROOT).as_posix()],
                                     cwd=ROOT, text=True)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


class BRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_default_three_batch_losses_gradients_and_updates_equal_original(self):
        old = historical_module(ROOT / 'scripts/train_refiner.py')
        initial = runner.LoopRefiner(8, dimension=16, heads=4, patch=4, image_size=32)
        torch.nn.init.normal_(initial.output_head[-1].weight, std=.02)
        fixtures = [(torch.randn(2, 8, 8, 8), torch.randn(2, 1, 32, 32),
                     torch.randint(0, 2, (2, 32, 32)).float(), torch.rand(2, 32, 32) > .1) for _ in range(3)]
        results = []
        for module in (old, runner):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            batches = []
            for features, draft, targets, valid in fixtures:
                optimizer.zero_grad(set_to_none=True)
                outputs = module.refine_steps(model, features, draft, 4)
                loss, steps, _, _, _ = module.refiner_objective(outputs, targets, valid, draft)
                loss.backward()
                grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
                batches.append((loss.detach().clone(), [v.item() for v in steps], grads))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            results.append((batches, copy.deepcopy(model.state_dict())))
        for original, current in zip(results[0][0], results[1][0]):
            self.assertTrue(torch.equal(original[0], current[0]))
            self.assertEqual(original[1], current[1])
            self.assertEqual(original[2].keys(), current[2].keys())
            self.assertTrue(all(torch.equal(original[2][key], current[2][key]) for key in original[2]))
        self.assertTrue(all(torch.equal(value, results[1][1][key]) for key, value in results[0][1].items()))

    def test_step_weights_normalization_validation_and_gradient(self):
        weights = runner.parse_step_weights('1,2,3,4')
        self.assertEqual(weights, [.1, .2, .3, .4])
        values = [torch.tensor(float(i), requires_grad=True) for i in (1, 2, 3, 4)]
        loss = runner.step_mean(values, weights)
        self.assertAlmostEqual(loss.item(), 3.)
        loss.backward()
        self.assertEqual([v.grad.item() for v in values], torch.tensor(weights).tolist())
        for text in ('', '0,0', '-1,2', 'nan,1', 'inf,1'):
            with self.assertRaises(Exception):
                runner.parse_step_weights(text)
        args = runner.build_parser().parse_args(['--step-weights', '1,2'])
        with self.assertRaisesRegex(ValueError, 'length'):
            runner.validate_step_training(args)
        args = runner.build_parser().parse_args(['--step-weights', '1,2,3,4', '--k-train-choices', '2,4'])
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            runner.validate_step_training(args)

    def test_nonzero_conv_head_step_weights_change_real_loss_and_gradients(self):
        initial = runner.LoopRefiner(8, dimension=16, heads=4, patch=16,
                                     image_size=32, head_type='conv_upsample')
        torch.nn.init.normal_(initial.output_head.final.weight, std=.05)
        features, draft = torch.randn(2, 8, 2, 2), torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32)).float()
        valid = torch.rand(2, 32, 32) > .1
        settings = [runner.build_parser().parse_args(arguments) for arguments in
                    ([], ['--step-weights', '0.1,0.2,0.3,0.4'])]
        results = []
        for args in settings:
            runner.validate_step_training(args)
            model = copy.deepcopy(initial).train()
            outputs = runner.refine_steps(model, features, draft, args.k_train)
            # This is the same objective call used by the actual BCE trainer.
            loss, steps, _, _, _ = runner.refiner_objective(
                outputs, targets, valid, draft, args.noadd_lambda,
                pos_weight=args.pos_weight, step_weights=args.step_weights)
            loss.backward()
            gradients = {name: parameter.grad.detach().clone()
                         for name, parameter in model.named_parameters() if parameter.grad is not None}
            results.append(dict(total=loss.item(), steps=[step.item() for step in steps],
                                gradients=gradients))
        uniform, weighted = results
        self.assertEqual(uniform['steps'], weighted['steps'])
        self.assertEqual(len(set(weighted['steps'])), 4, 'Nonzero head must produce four distinct losses')
        # Independent arithmetic is only an oracle, never the training loss.
        manual = sum(weight * loss for weight, loss in zip(settings[1].step_weights, weighted['steps']))
        self.assertAlmostEqual(weighted['total'], manual, delta=1e-6)
        self.assertNotAlmostEqual(weighted['total'], uniform['total'], delta=1e-6)
        different = [name for name, gradient in uniform['gradients'].items()
                     if not torch.equal(gradient, weighted['gradients'][name])]
        self.assertTrue(different, 'Step weights must change at least one parameter gradient')
        name = max(different, key=lambda key: abs(uniform['gradients'][key].norm().item() -
                                                 weighted['gradients'][key].norm().item()))
        print('WEIGHTED_STEP_GRADIENT ' + json.dumps(dict(
            per_step_losses=weighted['steps'], uniform_total=uniform['total'],
            weighted_total=weighted['total'], manual_weighted_total=manual,
            parameter=name, uniform_gradient_norm=uniform['gradients'][name].norm().item(),
            weighted_gradient_norm=weighted['gradients'][name].norm().item())), flush=True)

    def test_random_k_is_reproducible_without_global_rng_or_loader_changes(self):
        class Dataset:
            def __len__(self):
                return 20

            def __getitem__(self, index):
                return (np.zeros((3, 4, 4), np.float32), np.zeros((4, 4), np.float32),
                        np.ones((4, 4), bool), str(index), np.ones((4, 4), np.uint8))

        before = torch.get_rng_state().clone()
        choices = runner.parse_k_train_choices('2,3,4,5,6')
        sequences = []
        for seed in (0, 0, 1):
            generator = runner.make_k_generator(seed)
            sequences.append([runner.choose_training_k(choices, generator) for _ in range(100)])
        self.assertEqual(sequences[0], sequences[1])
        self.assertNotEqual(sequences[0], sequences[2])
        self.assertEqual(set(sequences[0]), set(choices))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        orders = []
        for random_k in (False, True):
            generator = runner.make_k_generator(0) if random_k else None
            loader = runner.make_loader(Dataset(), 3, 0, shuffle=True, seed=0)
            order = []
            for _, _, _, ids, _ in loader:
                order.extend(ids)
                if random_k:
                    runner.choose_training_k(choices, generator)
            orders.append(order)
        self.assertEqual(orders[0], orders[1])

    def test_random_k_rng_resume_and_ragged_step_means(self):
        loader = runner.make_loader([], 1, 0)
        generator = runner.make_k_generator(0)
        runner.choose_training_k([2, 3, 4, 5, 6], generator)
        state = runner.capture_rng(loader, generator)
        expected = [runner.choose_training_k([2, 3, 4, 5, 6], generator) for _ in range(20)]
        runner.restore_rng(state, loader, torch.device('cpu'), generator)
        self.assertEqual(expected, [runner.choose_training_k([2, 3, 4, 5, 6], generator) for _ in range(20)])
        values, counts = runner.ragged_step_means([dict(samples=2, step_losses=[1., 3.]),
                                                  dict(samples=1, step_losses=[5., 7., 9., 11.])], 'step_losses')
        self.assertEqual(counts, [3, 3, 1, 1])
        self.assertEqual(values, [7/3, 13/3, 9., 11.])

    def write_runs(self, experiments):
        for experiment in experiments:
            for seed in (0, 1, 2):
                report = fixture_report(seed, experiment != 'e0_v2')
                report['best_score'] = .9 - .1 * seed
                for group in report['evaluation']['groups'].values():
                    group['4']['scores'].update(road_iou=.6, cldice=.7, boundary_iou=.1)
                path = self.directory / 'results' / f'{experiment}_s{seed}' / 'metrics.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report))
        return self.directory / 'results'

    def test_missing_b_outputs_match_legacy_bytes(self):
        directory = self.write_runs((*aggregate.EXPERIMENTS, 'e6_convdraft'))
        (directory / 'summary.md').write_text('# Existing comparisons' + APPENDIX)
        old = historical_module(ROOT / 'analysis/aggregate_experiments.py')
        results = []
        for i, module in enumerate((old, aggregate)):
            out = self.directory / str(i)
            with patch.object(sys, 'argv', ['aggregate', '--results-dir', str(directory), '--out-dir', str(out)]), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            results.append([(out / name).read_bytes() for name in ('summary.md', 'summary.json')])
        self.assertEqual(results[0], results[1])

    def test_b_aggregates_use_val_sel_proxy_and_preserve_older_rows(self):
        directory = self.write_runs((*aggregate.EXPERIMENTS, 'e6_convdraft'))
        original = aggregate.aggregate_directory(directory)
        self.write_runs(('e7_stepw', 'e8_randk', 'e9_posw2'))
        updated = aggregate.aggregate_directory(directory)
        for key in ('rows', 'table1'):
            self.assertEqual(original[key], [r for r in updated[key] if not r['experiment'].startswith(('e7_', 'e8_', 'e9_'))])
        self.assertEqual(original['conv_draft_comparison'], updated['conv_draft_comparison'])
        row = next(r for r in updated['b_comparison']['rows'] if r['experiment']=='e7_stepw' and r['domain']=='Overall')
        self.assertAlmostEqual(row['Val_sel_proxy']['mean'], .8)
        self.assertAlmostEqual(row['Val_sel_proxy']['std'], .1)
        self.assertAlmostEqual(row['K4_minus_K1']['mean'], .006)
        (directory/'e8_randk_s2/metrics.json').unlink()
        row = next(r for r in aggregate.b_comparison(directory)['rows'] if r['experiment']=='e8_randk')
        self.assertEqual(row['status'], 'incomplete')

    def shell_run(self, arguments):
        capture = self.directory/'command.json'
        capture.unlink(missing_ok=True)
        fake = '''import json,os,sys
from pathlib import Path
args=sys.argv[1:];Path(os.environ['B_CAPTURE']).write_text(json.dumps(args))
out=Path(args[args.index('--out-dir')+1]);out.mkdir(parents=True,exist_ok=True)
(out/'metrics.json').write_text(json.dumps({'evaluation':{'groups':{'Overall':{'y0_draft':{'scores':{'A':.538246}}}}}}))
'''
        (self.directory/'env.sh').write_text('python() {\nif [[ "$1" == -u ]]; then "$B_PYTHON" -c ' +
            shlex.quote(fake) + ' "$@"; elif [[ $# == 1 ]]; then cat >/dev/null; else "$B_PYTHON" "$@"; fi\n}\ngit() { :; }\n')
        env = dict(os.environ, SGE_O_WORKDIR=str(self.directory), B_CAPTURE=str(capture), B_PYTHON=sys.executable)
        run = subprocess.run(['bash', str(JOB), *arguments], env=env, capture_output=True, text=True)
        return run, json.loads(capture.read_text()) if capture.exists() else None

    def test_qsub_accepts_only_variants_and_seeds_and_changes_one_setting(self):
        original = (ROOT/'jobs/a2_refiner_conv_draft.qsub').read_text()
        command = original.split('python -u scripts/train_refiner.py \\\n', 1)[1].split('# Read saved', 1)[0]
        base_tokens = shlex.split(command.replace('\\\n', ' '))
        names = dict(e7='e7_stepw', e8='e8_randk', e9='e9_posw2')
        for variant in names:
            for seed in ('0','1','2'):
                process, captured = self.shell_run([variant, seed])
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn('PASS B y0_draft', process.stdout)
                replaced = [seed if token=='$SEED' else 'results/exp/e6_convdraft_s'+seed if token=='$OUT_DIR'
                            else 'runs/exp/e6_convdraft_s'+seed if token=='$CKPT_DIR' else token for token in base_tokens]
                base = vars(runner.build_parser().parse_args(replaced))
                actual = vars(runner.build_parser().parse_args(captured[2:]))
                changed = {key for key in base if base[key] != actual[key]}
                expected = {'e7':'step_weights','e8':'k_train_choices','e9':'pos_weight'}[variant]
                self.assertEqual(changed, {'out_dir','ckpt_dir',expected})
                self.assertEqual(actual['out_dir'], Path(f'results/exp/{names[variant]}_s{seed}'))
        for arguments in ([], ['e7'], ['e0','0'], ['e7','3'], ['e8','01'], ['e9','0','extra']):
            process, captured = self.shell_run(arguments)
            self.assertEqual(process.returncode, 2)
            self.assertIsNone(captured)


if __name__ == '__main__':
    unittest.main()
