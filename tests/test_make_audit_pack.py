"""Deterministic sampling, anonymous metadata and blank thirty-row audit sheets."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from analysis import make_audit_pack as pack


class ManualAuditPackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_sampling_is_reproducible_unique_and_uses_required_filter(self):
        source=pack.ROOT/'results/failure/per_image.csv'
        first,n=pack.select_records(source);second,m=pack.select_records(source)
        self.assertEqual(first,second);self.assertEqual(n,m);self.assertEqual(len(first),30)
        self.assertEqual(len({r['id'] for r in first}),30)
        self.assertTrue(all(r['seed']=='0' and r['condition']=='ordinary' and r['domain']=='Urban' and float(r['delta_A4_y0'])<=-1e-4 for r in first))
        with source.open(newline='') as f:
            pool=[r for r in csv.DictReader(f) if r['seed']=='0' and r['condition']=='ordinary' and r['domain']=='Urban' and r['delta_A4_y0'] and float(r['delta_A4_y0'])<=-1e-4]
        expected=[pool[int(i)]['id'] for i in np.random.default_rng(0).choice(len(pool),30,replace=False)]
        self.assertEqual([r['id'] for r in first],expected)

    def test_panel_names_and_rendered_header_do_not_reveal_id_or_score(self):
        for i in range(1,31):self.assertEqual(pack.panel_name(i),f'audit_{i:02d}.png')
        image=self.root/'sensitive_image_id.png';label=self.root/'label.png'
        Image.new('RGB',(32,32)).save(image);Image.new('L',(32,32),1).save(label)
        masks={k:np.zeros((32,32),bool) for k in ('y0_draft','4')}
        path=self.root/pack.panel_name(1)
        with patch.object(pack.failure.ImageDraw,'Draw') as drawing:
            pack.render_anonymous(dict(image=str(image),label=str(label),id='sensitive_image_id',score=-.123456),masks,1,path)
            text=drawing.return_value.text.call_args.args[1]
            self.assertIn('Audit 01',text)
            self.assertNotIn('sensitive_image_id',text);self.assertNotIn('123456',text)
        self.assertTrue(path.is_file())

    def test_audit_sheet_has_thirty_blank_rows_and_private_key_is_exact(self):
        selected,_=pack.select_records(pack.ROOT/'results/failure/per_image.csv')
        pack.write_sheets(selected,self.root)
        with (self.root/'audit_sheet.csv').open(newline='') as f:
            reader=csv.DictReader(f);rows=list(reader);self.assertEqual(tuple(reader.fieldnames),pack.AUDIT_FIELDS)
        self.assertEqual(len(rows),30)
        for i,row in enumerate(rows,1):
            self.assertEqual(row['audit_no'],str(i));self.assertTrue(all(row[k]=='' for k in pack.AUDIT_FIELDS[1:]))
        with (self.root/'key.csv').open(newline='') as f:keys=list(csv.DictReader(f))
        for row,original in zip(keys,selected):
            self.assertEqual(row['id'],original['id']);self.assertEqual(row['delta_A4_y0'],original['delta_A4_y0'])
            self.assertEqual(row['added_pixels'],original['4_added']);self.assertEqual(row['added_precision'],original['4_added_precision'])

    def test_resume_preserves_completed_panels_and_rejects_changed_key(self):
        selected,_=pack.select_records(pack.ROOT/'results/failure/per_image.csv')
        out=self.root/'pack';out.mkdir();(out/'panels').mkdir();(out/'examples').mkdir()
        pack.write_sheets(selected,out)
        Image.new('RGB',(5120,1064)).save(out/'panels'/pack.panel_name(1))
        source=self.root/'example.png';Image.new('RGB',(16,16)).save(source)
        sources={name:source for name in pack.EXAMPLES}
        for name in sources:(out/'examples'/f'{name.replace("/","_")}.png').write_bytes(source.read_bytes())
        before=(out/'panels'/pack.panel_name(1)).read_bytes()
        self.assertEqual(pack.validate_resume(selected,out,sources),{pack.panel_name(1)})
        self.assertEqual((out/'panels'/pack.panel_name(1)).read_bytes(),before)
        key=out/'key.csv';key.write_text(key.read_text().replace(selected[0]['id'],'urban/changed',1))
        with self.assertRaisesRegex(ValueError,'key'):pack.validate_resume(selected,out,sources)


if __name__=='__main__':unittest.main()
