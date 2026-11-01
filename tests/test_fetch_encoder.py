"""Offline encoder acquisition, immutable identity and small-package checks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from submission import fetch_encoder as fetch
from scripts import package_submission as pack


class FetchEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.dest = self.root/'source', self.root/'destination'
        self.source.mkdir()
        self.contents = {name: ('fixture '+name).encode() for name in fetch.FILES}
        for name, data in self.contents.items():
            (self.source/name).write_bytes(data)
        self.hashes = {name: hashlib.sha256(self.contents[name]).hexdigest() for name in fetch.EXPECTED_SHA256}
        pinned = patch.dict(fetch.EXPECTED_SHA256, self.hashes, clear=True)
        pinned.start()
        self.addCleanup(pinned.stop)

    def archive(self, name='encoder.zip', corrupt=False):
        path = self.root/name
        with zipfile.ZipFile(path, 'w') as archive:
            for filename, data in self.contents.items():
                archive.writestr(filename, b'corrupt' if corrupt and filename == 'model.safetensors' else data)
        return path

    def assert_complete(self):
        for name, expected in self.contents.items():
            self.assertFalse((self.dest/name).is_symlink())
            self.assertEqual((self.dest/name).read_bytes(), expected)
        self.assertEqual(fetch.verify_encoder(self.dest), self.hashes)
        self.assertEqual(list(self.root.glob('.encoder-fetch-*')), [])

    def test_local_directory_dereferences_files(self):
        weight = self.source/'model.safetensors'
        target = self.root/'original.bin'
        weight.rename(target)
        weight.symlink_to(target)
        self.assertTrue(fetch.fetch_encoder(self.dest, from_dir=self.source))
        self.assert_complete()

    def test_file_url_download_and_unpack(self):
        self.assertTrue(fetch.fetch_encoder(self.dest, from_url=self.archive().as_uri()))
        self.assert_complete()

    def test_hash_mismatch_cleans_staged_copies_and_downloads(self):
        archive = self.archive(corrupt=True)
        (self.source/'model.safetensors').write_bytes(b'corrupt')
        self.dest.mkdir()
        (self.dest/'LICENSE.md').write_bytes(self.contents['LICENSE.md'])
        for source in ({'from_dir': self.source}, {'from_url': archive.as_uri()}):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                fetch.fetch_encoder(self.dest, **source)
            self.assertEqual(list(self.root.glob('.encoder-fetch-*')), [])
            self.assertEqual([p.name for p in self.dest.iterdir()], ['LICENSE.md'])
            self.assertEqual((self.dest/'LICENSE.md').read_bytes(), self.contents['LICENSE.md'])

    def test_valid_existing_files_skip_unavailable_source(self):
        fetch.fetch_encoder(self.dest, from_dir=self.source)
        before = {name: (self.dest/name).stat().st_mtime_ns for name in fetch.FILES}
        with patch.object(fetch, 'download_zip', side_effect=AssertionError('must skip download')):
            self.assertFalse(fetch.fetch_encoder(self.dest, from_url=(self.root/'missing.zip').as_uri()))
        self.assertEqual(before, {name: (self.dest/name).stat().st_mtime_ns for name in fetch.FILES})

    def test_prediction_verification_cache_and_modified_file(self):
        fetch.fetch_encoder(self.dest, from_dir=self.source)
        with patch.object(fetch, 'digest', side_effect=AssertionError('must reuse verified cache')):
            self.assertEqual(fetch.verify_encoder(self.dest), self.hashes)
        weight = self.dest/'model.safetensors'
        old = weight.stat()
        weight.write_bytes(b'X'*old.st_size)
        os.utime(weight, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch.*fetch_encoder.py'):
            fetch.verify_encoder(self.dest)
        with self.assertRaisesRegex(ValueError, 'fetch_encoder.py'):
            fetch.verify_encoder(self.root/'missing')

    def test_hf_calls_use_pinned_revision_without_network(self):
        def local_download(**kwargs):
            self.assertEqual(kwargs['repo_id'], fetch.REPO_ID)
            self.assertEqual(kwargs['revision'], 'ea8dc2863c51be0a264bab82070e3e8836b02d51')
            name = kwargs['filename']
            (Path(kwargs['local_dir'])/name).write_bytes(self.contents[name])
        with patch('huggingface_hub.hf_hub_download', side_effect=local_download) as mocked:
            fetch.fetch_encoder(self.dest, from_hf=True)
        self.assertEqual(mocked.call_count, 4)
        self.assert_complete()

    def test_unsafe_zip_is_rejected_and_cleaned(self):
        archive = self.root/'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('../config.json', b'unsafe')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            fetch.fetch_encoder(self.dest, from_url=archive.as_uri())
        self.assertFalse(self.dest.exists())
        self.assertFalse((self.root/'config.json').exists())
        self.assertEqual(list(self.root.glob('.encoder-fetch-*')), [])

    def test_prediction_rejects_bad_encoder_before_loading_models(self):
        from PIL import Image
        from submission import predict
        image = self.root/'tile.png'
        Image.new('RGB', (1024, 1024)).save(image)
        (self.source/'model.safetensors').write_bytes(b'corrupt')
        with patch.object(predict, 'load_baseline', side_effect=AssertionError('must not load models')):
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch.*fetch_encoder.py'):
                predict.main(['--input', str(image), '--output', str(self.root/'output'),
                              '--encoder-dir', str(self.source), '--checkpoint', 'unused',
                              '--baseline', 'unused', '--steps', '4', '--device', 'cpu'])
        self.assertFalse((self.root/'output').exists())

    def test_small_and_encoder_only_package_members(self):
        repo = self.root/'repo'
        sub = repo/'submission'
        sub.mkdir(parents=True)
        (sub/'fetch_encoder.py').write_text('# fixture\n')
        (sub/'README.md').write_text('Fixture documentation\n')
        (repo/'requirements.txt').write_text('# fixture\n')
        for name in ('model.safetensors', 'baseline.safetensors'):
            (sub/name).write_bytes(b'fixture')
        (sub/'encoder_sha256.json').write_text(json.dumps(dict(repo_id=fetch.REPO_ID, revision=fetch.REVISION, sha256=self.hashes)))
        with patch.object(pack, 'ROOT', repo):
            small = pack.package(self.root/'small', self.source, no_encoder=True)
            encoder = pack.package(self.root/'encoder', self.source, encoder_only=True)
        with zipfile.ZipFile(small) as zipped:
            names = set(zipped.namelist())
            self.assertTrue({'submission/fetch_encoder.py', 'submission/encoder_sha256.json', 'submission/encoder/LICENSE.md'} <= names)
            self.assertFalse(any('submission/encoder/'+name in names for name in fetch.EXPECTED_SHA256))
        with zipfile.ZipFile(encoder) as zipped:
            self.assertEqual(set(zipped.namelist()), set(fetch.FILES))
        self.assertLess(small.stat().st_size, 95*1024**2)


if __name__ == '__main__':
    unittest.main()
