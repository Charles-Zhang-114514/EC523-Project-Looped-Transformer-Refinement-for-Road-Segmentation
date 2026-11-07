"""Geometric connection proxy, colors, exact official aggregation and domain bounds."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from analysis import failure_analysis as audit
from analysis import domain_gating_bound as bound


class FailureAnalysisTests(unittest.TestCase):
    def test_correct_false_and_absent_merge(self):
        base=np.zeros((12,20),bool);base[5:7,2:6]=True;base[5:7,12:16]=True
        pred=base.copy();pred[5:7,6:12]=True
        valid=np.ones_like(base)
        self.assertEqual(audit.connections(base,pred,pred,valid),dict(false_connection=0,correct_connection=1))
        self.assertEqual(audit.connections(base,pred,base,valid),dict(false_connection=1,correct_connection=0))
        self.assertEqual(audit.connections(base,base,base,valid),dict(false_connection=0,correct_connection=0))
        valid[:,8]=False
        self.assertEqual(audit.connections(base,pred,base,valid),dict(false_connection=0,correct_connection=0))

    def test_difference_colors_and_ignore(self):
        base=np.array([[0,0,1,1,0,0]],bool)
        pred=np.array([[1,1,0,0,0,1]],bool)
        truth=np.array([[1,0,1,0,0,1]],bool)
        valid=np.array([[1,1,1,1,1,0]],bool)
        expected=[(0,255,0),(255,0,0),(0,0,255),(0,255,255),(128,128,128),(128,128,128)]
        self.assertEqual(audit.difference_colors(base,pred,truth,valid)[0].tolist(),list(map(list,expected)))

    def test_per_image_counts_sum_matches_official_and_empty_gt_is_separate(self):
        audit.initialize_worker(os.environ['EC523_STARTER_DIR']);metrics=audit._METRICS
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            label=Path(directory)/'mask.png'
            labels=np.ones((48,48),np.uint8);labels[20:24,4:40]=3
            Image.fromarray(labels).save(label)
            gt=labels==3;base=gt.copy();base[:,20:24]=False
            row=dict(id='urban/synthetic',domain='urban',condition='ordinary',label=str(label))
            record,_,_,_=audit.process_image(row,dict(y0_draft=base,**{'1':gt,'4':gt}),0)
            pooled=audit.aggregate([record,record],metrics)
            seeds=audit.seed_summary({'ordinary/s0':{'Overall':pooled},'ordinary/s1':{'Overall':pooled}})
            self.assertEqual(seeds['ordinary/Overall']['false_connection']['sample_std'],0)
            self.assertEqual(seeds['ordinary/Overall']['better_fraction']['mean'],pooled['delta_A4_y0']['better'])
            for key,mask in [('y0_draft',base),('1',gt),('4',gt)]:
                expected=metrics.image_counts(mask.astype(np.float32),labels)
                self.assertEqual(pooled['groups'][key]['counts'],asdict(expected+expected))
            labels[:]=1;Image.fromarray(labels).save(label)
            empty,_,_,_=audit.process_image(row,dict(y0_draft=base,**{'1':gt,'4':gt}),0)
            self.assertIsNone(empty['A']['4']);self.assertIsNone(empty['delta_A4_y0'])
            self.assertEqual(audit.aggregate([empty],metrics)['delta_A4_y0']['applicable_images'],0)

    def test_domain_count_sum_and_equal_threshold_identity(self):
        metrics=bound.official_metrics(os.environ['EC523_STARTER_DIR'])
        source=json.loads((bound.ROOT/'results/infer_probe/metrics.json').read_text())
        groups=source['gating']['e0_v2_s0']['Val-rep']['groups']
        for k in ('1','4'):
            value=bound.combine(groups['Urban'][k+':0.50'],groups['Rural'][k+':0.50'],metrics)
            self.assertEqual(value['counts'],groups['Overall'][k+':0.50']['counts'])
            self.assertEqual(value['scores'],groups['Overall'][k+':0.50']['scores'])

    def test_selection_reproducible_without_replacement_and_empty_pools(self):
        record=dict(seed=0,condition='ordinary',domain='Urban',id='urban/a',kind='ordinary',
            delta_A4_y0=.01,delta_A8_K4=-.02,gt_has_road=True,
            steps={'4':dict(false_connection=0,correct_connection=1,added=10)})
        records=[dict(record,id='urban/'+str(i),delta_A4_y0=i/100+.001) for i in range(8)]
        a,overview=audit.select_cases(records);b,_=audit.select_cases(records)
        self.assertEqual(a,b)
        self.assertEqual(overview['clean_Rural_negative']['selected'],0)
        for category in overview:
            keys=[(r['record']['condition'],r['record']['id']) for r in a if r['category']==category]
            self.assertEqual(len(keys),len(set(keys)))


if __name__=='__main__':unittest.main()
