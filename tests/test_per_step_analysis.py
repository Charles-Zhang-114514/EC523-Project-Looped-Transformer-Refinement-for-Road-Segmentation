"""Region geometry, per-update accounting and zero/sub-threshold activity."""
import unittest
import os
from dataclasses import asdict

import numpy as np
from analysis import per_step_analysis as step


class PerStepAnalysisTests(unittest.TestCase):
    def labels(self):
        labels=np.ones((64,64),np.uint8);labels[20:44,20:44]=3;labels[0:4,0:4]=0
        return labels

    def test_regions_partition_valid_pixels_and_distance_edges(self):
        labels=self.labels();r=step.assign_regions(labels)
        self.assertTrue(np.array_equal(r>0,labels!=0))
        self.assertEqual(set(np.unique(r)),{0,1,2,3,4})
        self.assertEqual(r[30,20],1);self.assertEqual(r[30,17],1)
        self.assertEqual(r[30,16],3);self.assertEqual(r[30,4],3);self.assertEqual(r[30,3],4)
        self.assertEqual(r[30,23],1);self.assertEqual(r[30,24],2)
        coverage=sum((r==i).astype(np.uint8) for i in range(1,5))
        np.testing.assert_array_equal(coverage,(labels!=0).astype(np.uint8))

    def test_region_additions_and_deletions_sum_to_totals(self):
        labels=self.labels();r=step.assign_regions(labels)
        rng=np.random.default_rng(0);previous=rng.random(labels.shape).astype(np.float32);current=rng.random(labels.shape).astype(np.float32)
        value=step.step_changes(previous,current,labels,r)
        self.assertEqual(sum(v['added'] for v in value['regions'].values()),value['added'])
        self.assertEqual(sum(v['deleted'] for v in value['regions'].values()),value['deleted'])
        self.assertEqual(value['added'],value['added_correct']+value['added_wrong'])
        self.assertEqual(value['deleted'],value['deleted_correct']+value['deleted_wrong'])

    def test_identical_probabilities_have_zero_changes(self):
        labels=self.labels();r=step.assign_regions(labels)
        p=np.random.default_rng(1).random(labels.shape).astype(np.float32)
        value=step.step_changes(p,p.copy(),labels,r)
        self.assertEqual(value['added'],0);self.assertEqual(value['deleted'],0)
        for v in value['regions'].values():
            for name in ('sum_abs_probability_change','change_over_005','no_flip_sum_abs','no_flip_over_005',
                         'below_threshold_sum_abs','below_threshold_over_005'):self.assertEqual(v[name],0)
        metrics=step.runner.probe.official_metrics(os.environ['EC523_STARTER_DIR'])
        counts=asdict(metrics.image_counts(p,labels))
        bucket=step.empty_bucket(metrics,1)
        bucket.update(images=1,valid_pixels=int((labels!=0).sum()),y0_counts=counts)
        bucket['steps']['1']=dict(**value,counts=counts,tracking=dict.fromkeys(step.TRACK,0))
        result=step.finalize(bucket,metrics)
        self.assertTrue(all(v==0 for v in result['steps']['1']['metric_changes'].values()))

    def test_empty_truth_is_entirely_R4_and_tracks_additions(self):
        labels=np.ones((32,32),np.uint8);labels[:2,:2]=0
        r=step.assign_regions(labels);self.assertFalse(np.isin(r,[1,2,3]).any())
        np.testing.assert_array_equal(r==4,labels!=0)
        out=step.step_changes(np.zeros(labels.shape,np.float32),np.ones(labels.shape,np.float32),labels,r)
        self.assertEqual(out['no_gt_added'],1020);self.assertEqual(out['regions']['R4']['added'],1020)

    def test_probability_motion_without_threshold_crossing_is_retained(self):
        labels=self.labels();r=step.assign_regions(labels)
        before=np.full(labels.shape,.1,np.float32);after=np.full(labels.shape,.2,np.float32)
        out=step.step_changes(before,after,labels,r)
        self.assertEqual(out['added'],0)
        for v in out['regions'].values():
            self.assertAlmostEqual(v['sum_abs_probability_change']/v['pixels'],.1,places=6)
            self.assertEqual(v['change_over_005'],v['pixels'])
            self.assertEqual(v['no_flip_over_005'],v['pixels'])
            self.assertEqual(v['below_threshold_over_005'],v['pixels'])


if __name__=='__main__':unittest.main()
