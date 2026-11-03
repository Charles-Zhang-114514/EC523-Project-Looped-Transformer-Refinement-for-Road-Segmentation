"""Official manifest compatibility, pooled counts and valid rectangle splits."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.build_eval_manifest import build_manifest, write_manifest
from scripts import eval_stress as stress


class StressEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.starter = Path(os.environ['EC523_STARTER_DIR'])

    def test_manifest_loads_in_official_common_without_label_remapping(self):
        split = {'Val-rep': {'Urban': ['b.png','a.png'], 'Rural':['a.png']}}
        path = self.root/'split.json'
        path.write_text(json.dumps(split))
        for domain,names in split['Val-rep'].items():
            for name in names:
                base=self.root/'data/Val'/domain
                (base/'images_png').mkdir(parents=True,exist_ok=True)
                (base/'masks_png').mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(12,12)).save(base/'images_png'/name)
                Image.fromarray(np.full((12,12),7,np.uint8)).save(base/'masks_png'/name)
        rows=build_manifest(path,self.root/'data')
        manifest=write_manifest(self.root/'ordinary.json',rows)
        loaded=stress.official_module('common',self.starter).load_manifest(manifest)
        self.assertEqual([r['id'] for r in loaded],['urban/a','urban/b','rural/a'])
        self.assertTrue(all(r['condition']=='ordinary' and Path(r['label']).is_absolute() for r in loaded))
        self.assertTrue(np.all(np.asarray(Image.open(loaded[0]['label']))==7))
        self.assertEqual(len(build_manifest(path,self.root/'data',1)),2)

    def test_corruption_counts_partition_overall_and_empty_subsets(self):
        stress.initialize_worker(self.starter)
        metrics=stress._METRICS
        buckets=stress.empty_buckets(['y0_draft','1','4'],metrics)
        paired={'stress_test':stress.empty_buckets(['y0_draft','1','4'],metrics)}
        assignments={'stress_test':{str(i):kind for i,kind in enumerate(stress.KINDS)}}
        for index,kind in enumerate(stress.KINDS):
            labels=np.ones((16,16),np.uint8)
            labels[6:9,2:14]=3
            labels[0,:]=0
            path=self.root/f'label{index}.png'
            Image.fromarray(labels).save(path)
            draft=labels==3
            draft[:,7:9]=False
            predicted=labels==3
            row=dict(id=str(index),label=str(path),domain='urban' if index<2 else 'rural',
                     corruption=dict(kind=kind,rectangle_xywh=[2,3,8,8]))
            value=stress.analyze_image(row,{'y0_draft':draft,'1':predicted,'4':predicted})
            stress.add_result(buckets,value)
            ordinary=(value[0],None,value[2],value[3],value[4])
            stress.add_paired_clean(paired,assignments,ordinary)
        report=stress.finalize(buckets,metrics,True)
        paired_report=stress.finalize(paired['stress_test'],metrics,True)
        for group in stress.GROUPS:
            for key in ('y0_draft','1','4'):
                self.assertEqual(report[group][key]['counts'],paired_report[group][key]['counts'])
        for key in ('y0_draft','1','4'):
            total=metrics.Counts()
            for kind in stress.KINDS:
                total=total+metrics.Counts(**report[kind][key]['counts'])
            self.assertEqual(total,metrics.Counts(**report['Overall'][key]['counts']))
        self.assertIsNone(report['Rural/shadow']['1']['scores'])
        pooled=stress.pooled_models([{'models':{'0':{'groups':report}}}]*2,[0],metrics)
        for name,count in report['Overall']['1']['counts'].items():
            self.assertEqual(pooled['models']['0']['groups']['Overall']['1']['counts'][name],2*count)

    def test_rectangle_inside_outside_partition_and_correct_pixels(self):
        draft=np.zeros((8,8),bool)
        prediction=np.zeros_like(draft)
        prediction[1,1]=prediction[2,2]=prediction[5,5]=prediction[6,6]=prediction[0,0]=True
        truth=np.zeros_like(draft)
        truth[1,1]=truth[5,5]=True
        valid=np.ones_like(draft)
        valid[0,0]=False
        result=stress.rectangle_changes(draft,prediction,truth,valid,[1,1,3,3])
        self.assertEqual(result,dict(inside_added=2,inside_correct=1,outside_added=2,outside_correct=1))
        self.assertEqual(result['inside_added']+result['outside_added'],int((~draft & prediction & valid).sum()))
        with self.assertRaises(ValueError):
            stress.rectangle_changes(draft,prediction,truth,valid,[7,7,2,2])


if __name__=='__main__':
    unittest.main()
