"""Confidence partitions, disjoint threshold selection and residual semantics."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from analysis.gating_sweep import gated_mask, confidence_strata, choose_tau, stress_supplement
from looproad.models.refiner import LoopRefiner
from scripts import train_refiner as runner


class InferProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_half_gate_equals_native_and_higher_gates_only_remove_additions(self):
        rng=np.random.default_rng(12)
        base=rng.random((32,32))>=.5
        probability=rng.random((32,32)).astype(np.float32)
        native=probability>=.5
        self.assertTrue(np.array_equal(gated_mask(base,probability,.5),native))
        for tau in (.55,.8,.95):
            mask=gated_mask(base,probability,tau)
            self.assertFalse((mask & ~native).any())
            self.assertTrue(np.array_equal(mask & base,native & base))
            self.assertTrue(np.array_equal(base & ~mask,base & ~native))

    def test_confidence_bins_include_boundaries_and_partition_valid_additions(self):
        probability=np.array([[.5,.6,.7,.8,.9,1.,.49]],np.float32)
        base=np.zeros_like(probability,dtype=bool)
        valid=np.ones_like(base)
        truth=np.array([[1,0,1,0,1,0,0]],bool)
        bands=confidence_strata(base,probability,truth,valid)
        self.assertEqual([b['pixels'] for b in bands],[1,1,1,1,2])
        self.assertEqual(sum(b['correct'] for b in bands),3)
        valid[0,0]=False
        base[0,1]=True
        self.assertEqual(sum(b['pixels'] for b in confidence_strata(base,probability,truth,valid)),4)

    def test_tau_selection_uses_only_val_sel_weighting_and_lowest_tie(self):
        values={f'{k}:{tau}':{'scores':{'quality':score}} for k in (1,4)
                for tau,score in (('0.50',.1),('0.80',.2))}
        report=dict(split='Val-sel',groups={'Overall':values})
        selected=choose_tau(report,[.5,.8])
        self.assertEqual(selected['tau'],'0.80')
        self.assertAlmostEqual(selected['proxy'],.33*.2+.67*.2)
        values['1:0.50']['scores']['quality']=.2
        values['4:0.50']['scores']['quality']=.2
        self.assertEqual(choose_tau(report,[.8,.5])['tau'],'0.50')
        with self.assertRaisesRegex(ValueError,'Val-sel'):
            choose_tau(dict(report,split='Val-rep'),[.5,.8])

    def test_new_intervention_keeps_residual_memory_and_fixed_answer_tokens(self):
        self.assertEqual(runner.parse_infer_ablations('y0_answer_keep_residual'), ['y0_answer_keep_residual'])
        torch.manual_seed(92)
        model=LoopRefiner(8,16,heads=4,patch=4,image_size=16).eval()
        torch.nn.init.normal_(model.output_head[-1].weight,std=.2)
        features=torch.randn(1,8,4,4)
        draft=torch.randn(1,1,16,16)
        with torch.no_grad():
            expected=runner.refine_steps(model,features,draft,4)
        observed=[answer.clone() for _,answer,_,_ in runner.inference_steps(
            model,features,draft,[1,2,4],torch.device('cpu'),'y0_answer_keep_residual')]
        self.assertTrue(torch.equal(observed[0],expected[0]))
        with torch.no_grad():
            state,tokens=model.init(features,draft)
            fixed=model.answer_tokens(draft)
            current=draft
            reference=[]
            for _ in range(4):
                state=model.core(state+tokens+fixed+model.position_encoding)
                current=current+torch.nn.functional.fold(
                    model.output_head(state).transpose(1,2),output_size=(16,16),kernel_size=4,stride=4)
                reference.append(current.clone())
        for observed_answer,index in zip(observed,(0,1,3)):
            self.assertTrue(torch.equal(observed_answer,reference[index]))
        self.assertFalse(model.answer_projection._forward_hooks)
        normal=[y.clone() for _,y,_,_ in runner.inference_steps(model,features,draft,[1,4],torch.device('cpu'))]
        self.assertTrue(torch.equal(normal[0],expected[0]))
        self.assertTrue(torch.equal(normal[1],expected[3]))

    def test_new_feedback_hook_is_removed_when_generator_is_closed(self):
        model=LoopRefiner(8,16,heads=4,patch=4,image_size=16)
        generator=runner.inference_steps(model,torch.randn(1,8,4,4),torch.randn(1,1,16,16),
            [1,4],torch.device('cpu'),'y0_answer_keep_residual')
        next(generator)
        self.assertTrue(model.answer_projection._forward_hooks)
        generator.close()
        self.assertFalse(model.answer_projection._forward_hooks)

    def test_stress_rows_copy_stored_pooled_scores_and_matched_clean(self):
        path=Path(runner.ROOT/'results/stress/v2/metrics.json')
        data=json.loads(path.read_text())
        rows=stress_supplement(path)['rows']
        self.assertEqual(len(rows),12)
        item=next(r for r in rows if r['condition']=='shadow' and r['domain']=='Urban')
        source=data['combined_stress']['models']['0']['groups']['Urban/shadow']['y0_draft']['scores']
        clean=data['paired_clean']['stress_pooled']['models']['0']['groups']['Urban/shadow']['y0_draft']['scores']
        self.assertEqual(item['scores']['quality'],source['quality'])
        self.assertEqual(item['drop']['quality'],clean['quality']-source['quality'])
        self.assertEqual(item['break_even'],source['road_iou']/(1+source['road_iou']))


if __name__=='__main__':
    unittest.main()
