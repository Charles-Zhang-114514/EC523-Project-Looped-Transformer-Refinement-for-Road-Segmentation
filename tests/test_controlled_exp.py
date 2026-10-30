"""Controlled split isolation, selection, extra diagnostics and reporting checks."""
import argparse
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.make_val_split import build_split, validate_split
from scripts import train_refiner as runner
from looproad.models.refiner import LoopRefiner
from analysis.aggregate_experiments import (
    aggregate_directory, mean_row, read_run, table1_rows, COLUMNS, EXPERIMENTS,
)


class FakeDataset:
    def __init__(self, names):
        self.pairs = [(Path('images')/n, Path('masks')/n) for n in sorted(names)]

    def __len__(self):
        return len(self.pairs)


class ControlledExperimentTests(unittest.TestCase):
    def names(self):
        return {'Urban': [f'{i}.png' for i in range(7)], 'Rural': [f'{i}.png' for i in range(8)]}

    def test_split_repeatability_partition_and_odd_counts(self):
        names = self.names()
        split = build_split(names)
        self.assertEqual(split, build_split({k:list(reversed(v)) for k,v in names.items()}))
        validate_split(split, names)
        for domain in names:
            a, b = set(split['Val-sel'][domain]), set(split['Val-rep'][domain])
            self.assertFalse(a & b)
            self.assertEqual(a | b, set(names[domain]))
        self.assertEqual(split['counts'], {'Val-sel': {'Urban':3,'Rural':4}, 'Val-rep':{'Urban':4,'Rural':4}})

    def test_committed_split_matches_full_readonly_val(self):
        names = {d:[a.name for a,_ in runner.probe.RoadDataset('Val',d).pairs] for d in runner.DOMAINS}
        saved = json.loads((runner.ROOT/'configs/val_split_seed0.json').read_text())
        validate_split(saved, names)
        self.assertEqual(saved, build_split(names, 0))
        self.assertEqual(saved['counts'], {'Val-sel':{'Urban':338,'Rural':496},'Val-rep':{'Urban':339,'Rural':496}})

    def test_selection_subset_and_report_membership_are_disjoint(self):
        names = self.names()
        split = build_split(names)
        with patch.object(runner.probe, 'RoadDataset', side_effect=lambda split_name,domain: FakeDataset(names[domain])):
            selected = runner.protocol_datasets(split, 'Val-sel', 2)
            selected_again = runner.protocol_datasets(split, 'Val-sel', 2)
            reported = runner.protocol_datasets(split, 'Val-rep')
        for d in runner.DOMAINS:
            a = [p.name for p,_ in selected[d].pairs]
            b = [p.name for p,_ in reported[d].pairs]
            self.assertEqual(a, [p.name for p,_ in selected_again[d].pairs])
            self.assertLessEqual(set(a), set(split['Val-sel'][d]))
            self.assertEqual(b, split['Val-rep'][d])
            self.assertFalse(set(a) & set(b))

    def test_selection_parser_weights_and_invalid_inputs(self):
        weights = runner.parse_selection_ks('4:0.5,1:0.25,2:0.25')
        self.assertEqual(list(weights), ['1','2','4'])
        scores = {'1':.1,'2':.2,'4':.3}
        self.assertEqual(runner.selection_score(scores,weights), runner.selection_score(scores))
        self.assertEqual(runner.selection_score({'1':.42},runner.parse_selection_ks('1:1.0')), .42)
        for text in ('0:1', '1:0', '1:-1', '1:0.5', '1:nan', '1:inf', '1:0.5,1:0.5', 'bad'):
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_selection_ks(text)

    def test_default_arguments_preserve_legacy_settings_and_flags(self):
        args = runner.build_parser().parse_args([])
        self.assertIsNone(args.val_split)
        self.assertFalse(args.eval_extra or args.reset_state or args.no_mask_feedback)
        self.assertEqual((args.epochs,args.k_train,args.batch_size,args.head_type,args.select_every),(20,4,4,'patch_linear',0))
        self.assertEqual(args.selection_ks, {'1':.25,'2':.25,'4':.5})
        flags=runner.build_parser().parse_args(['--reset-state','--no-mask-feedback'])
        self.assertTrue(flags.reset_state and flags.no_mask_feedback)

    def test_single_step_training_calls_core_once_and_updates_head(self):
        torch.manual_seed(0)
        model=LoopRefiner(8,16,heads=4,image_size=32,head_type='conv_upsample')
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        calls=[]
        hook=model.core.register_forward_hook(lambda *args: calls.append(1))
        before=model.output_head.final.weight.detach().clone()
        outputs=runner.refine_steps(model,torch.randn(2,8,2,2),torch.randn(2,1,32,32),1)
        loss,steps=runner.deep_supervision(outputs,torch.ones(2,32,32),torch.ones(2,32,32,dtype=torch.bool))
        loss.backward()
        optimizer.step()
        hook.remove()
        self.assertEqual(len(steps),1)
        self.assertEqual(len(calls),1)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before,model.output_head.final.weight))

    def test_extra_counts_reuse_legacy_scores_and_exact_grid_definition(self):
        runner.initialize_eval_worker(os.environ['EC523_STARTER_DIR'])
        labels=np.ones((64,64),dtype=np.uint8)
        labels[40:43,5:60]=3
        draft=labels==3
        predicted=draft.copy()
        predicted[:16,:16]=True
        predictions={'y0_draft':draft,'1':predicted}
        plain=runner.score_image('Urban','test',predictions,labels,True)[2]
        extra=runner.score_image('Urban','test',predictions,labels,True,True)[2]
        for key in plain:
            self.assertEqual(plain[key],{k:v for k,v in extra[key].items() if k!='extra'})
        e=extra['1']['extra']
        self.assertEqual(e['changes'],dict(added_correct=0,added_wrong=256,deleted_correct=0,deleted_wrong=0))
        self.assertEqual(e['new_fragments'],{'count':1,'aligned_edges':4})

    def test_aggregation_missing_runs_and_three_seed_std(self):
        self.assertEqual(read_run('incomplete_smoke', Path('absent'))['status'], 'missing')
        with patch.object(Path,'glob',return_value=[]), patch.object(Path,'read_text',side_effect=FileNotFoundError):
            result=aggregate_directory(Path('absent'))
        self.assertEqual(len(result['rows']),16)
        self.assertTrue(all(r['status']=='missing' for r in result['rows'][:12]))
        self.assertTrue(all(r['status']=='incomplete' for r in result['rows'][12:]))
        self.assertTrue(all(r['status']=='incomplete' for r in result['table1']))
        rows=[dict(experiment='e0_v2',seed=i,status='complete',**{name:float(i) for name in COLUMNS[3:]}) for i in (0,1,2)]
        mean=mean_row(rows)
        self.assertEqual(mean['proxy'],{'mean':1.0,'std':1.0})

    def test_each_experiment_uses_three_seed_sample_standard_deviation(self):
        for experiment in EXPERIMENTS:
            rows = [dict(experiment=experiment, seed=seed, status='complete',
                         **{name: value for name in COLUMNS[3:]})
                    for seed, value in enumerate((1.0, 3.0, 5.0))]
            result = mean_row(rows, experiment)
            self.assertEqual(result['experiment'], f'{experiment} mean ± std (3 seeds)')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed_seeds'], 3)
            for name in COLUMNS[3:]:
                self.assertEqual(result[name], {'mean': 3.0, 'std': 2.0})

    def test_experiment_with_missing_seed_is_incomplete_without_partial_mean(self):
        for experiment in EXPERIMENTS:
            rows = [dict(experiment=experiment, seed=seed, status='complete',
                         **{name: float(seed) for name in COLUMNS[3:]}) for seed in (0, 2)]
            rows.append(dict(experiment=experiment, seed=1, status='missing'))
            result = mean_row(rows, experiment)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['completed_seeds'], 2)
            self.assertNotIn('proxy', result)

    def test_table1_uses_seed_order_means_and_shared_draft_reference(self):
        rows = [dict(experiment=experiment, seed=seed, status='complete',
                     Urban_y0_A=.6, Rural_y0_A=.4,
                     **{name: float(seed + 1) for name in COLUMNS[3:]})
                for experiment in EXPERIMENTS for seed in (2, 0, 1)]
        aggregates = [mean_row(rows, experiment) for experiment in EXPERIMENTS]
        table = table1_rows(rows, aggregates)
        self.assertEqual(len(table), 5)
        baseline = table[0]
        for key in ('A_K1', 'A_K2', 'A_K4', 'A_K6', 'A_K8', 'proxy'):
            self.assertEqual(baseline[key], {'mean': 2.0, 'std': 1.0})
        self.assertEqual((baseline['Urban_A_K4'], baseline['Rural_A_K4']), (.6, .4))
        self.assertIsNone(baseline['correct_added_K1'])
        self.assertEqual(baseline['new_break_K4'], 0.0)
        for row in table[1:]:
            self.assertEqual(row['best_epochs'], [1.0, 2.0, 3.0])
            self.assertEqual(row['proxy'], {'mean': 2.0, 'std': 1.0})
            self.assertEqual(row['gap_repair_K4'], 2.0)


if __name__ == '__main__':
    unittest.main()
