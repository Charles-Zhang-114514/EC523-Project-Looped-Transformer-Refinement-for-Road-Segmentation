#!/usr/bin/env python3
"""Create a standalone, dereferenced inference directory and ZIP allowlist.

Nothing is installed or downloaded. Raw data, developer environments, course
code, training checkpoints and runtime outputs can never enter the allowlist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submission.fetch_encoder import EXPECTED_SHA256, REPO_ID, REVISION, verify_encoder

ENCODER_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def payload_files(encoder_dir, *, no_encoder=False, encoder_only=False):
    """Select sources explicitly; never recursively copy a working directory."""
    if no_encoder and encoder_only:
        raise ValueError('--no-encoder and --encoder-only are mutually exclusive')
    encoder_dir = Path(encoder_dir)
    selected = {Path('requirements.txt'): ROOT/'requirements.txt',
                Path('submission/README.md'): ROOT/'submission/README.md',
                Path('submission/encoder_sha256.json'): ROOT/'submission/encoder_sha256.json'}
    manifest = json.loads((ROOT/'submission/encoder_sha256.json').read_text())
    if manifest != dict(repo_id=REPO_ID, revision=REVISION, sha256=EXPECTED_SHA256):
        raise ValueError('Encoder SHA-256 manifest does not match the pinned fetch script')
    for directory in (ROOT/'submission', ROOT/'src/looproad'):
        paths = directory.glob('*.py') if directory.name == 'submission' else directory.rglob('*.py')
        for path in paths:
            if '__pycache__' not in path.parts:
                selected[path.relative_to(ROOT)] = path
    for name in ('model.safetensors', 'baseline.safetensors'):
        selected[Path('submission')/name] = ROOT/'submission'/name
    if not no_encoder:
        verify_encoder(encoder_dir, use_cache=False, write_cache=False)
        for name in ENCODER_FILES:
            selected[Path('submission/encoder')/name] = encoder_dir/name
    licenses = [encoder_dir/'LICENSE.md']
    for path in licenses:
        selected[Path('submission/encoder')/path.name] = path
    if encoder_only:
        selected = {Path(name): encoder_dir/name for name in (*ENCODER_FILES, 'LICENSE.md')}
    for relative, source in selected.items():
        roots = (*ENCODER_FILES, 'LICENSE.md') if encoder_only else ('submission', 'src', 'requirements.txt')
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in roots:
            raise ValueError('Unsafe package member')
        if not source.is_file():
            raise FileNotFoundError(f'Missing package file: {relative}')
        if source.suffix in ('.py', '.md', '.txt', '.json') or source.name.upper().startswith('LICENSE'):
            # Reject personal account IDs and cluster/home absolute paths.
            text = source.read_text(encoding='utf-8')
            account = os.environ.get('USER', '')
            personal_account = (account not in ('', 'root', 'nobody') and
                                re.search(r'\b'+re.escape(account)+r'\b', text))
            if personal_account or re.search(r'/(?:project[a-z]+|usr\d+|home)/', text):
                raise ValueError(f'Personal path/account found in package text: {relative}')
    return dict(sorted(selected.items())), licenses


def package(out, encoder_dir, *, no_encoder=False, encoder_only=False):
    out = Path(out).resolve()
    if out.is_relative_to(ROOT):
        if not out.is_relative_to(ROOT/'runs'):
            raise ValueError('--out must be outside the repository or under ignored runs/')
        ignored = subprocess.run(['git', 'check-ignore', str(out.relative_to(ROOT))], cwd=ROOT, capture_output=True)
        if ignored.returncode:
            raise ValueError('Output under runs/ must be ignored by git')
    archive = Path(str(out)+'.zip')
    if out.exists() or archive.exists():
        raise FileExistsError('Choose a new package output directory and ZIP filename')
    files, licenses = payload_files(encoder_dir, no_encoder=no_encoder, encoder_only=encoder_only)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = sum(source.stat().st_size for source in files.values())
    if shutil.disk_usage(out.parent).free < 2*size + 100*1024**2:
        raise OSError('Insufficient disk space for the copied payload and ZIP')
    out.mkdir()
    for relative, source in files.items():
        destination = out/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.resolve(strict=True), destination)
        destination.chmod(0o644)
        # Stable ZIP metadata: no developer paths, permissions or timestamps.
        os.utime(destination, (1577836800, 1577836800))
        if destination.is_symlink() or digest(destination) != digest(source):
            raise ValueError(f'Copy verification failed: {relative}')
        print(f'FILE {relative.as_posix()} bytes={destination.stat().st_size} sha256={digest(destination)}', flush=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for relative in files:
            zipped.write(out/relative, arcname=relative.as_posix())
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            relative = Path(item.filename)
            if not stat.S_ISREG(item.external_attr >> 16) or relative not in files:
                raise ValueError('ZIP contains a symlink or non-allowlisted entry')
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('ZIP contains an unsafe path')
    if no_encoder and archive.stat().st_size >= 95*1024**2:
        archive.unlink()
        raise ValueError('Small submission ZIP must be strictly below 95 MiB')
    print('LICENSE ' + (','.join(p.name for p in licenses) if licenses else 'No local LICENSE file; subject to the DINOv3 License'), flush=True)
    print(f'ZIP {archive.name} bytes={archive.stat().st_size} sha256={digest(archive)} files={len(files)} payload_bytes={size}', flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--encoder-dir', type=Path, default=ROOT/'submission/encoder')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--no-encoder', action='store_true', help='Small Gradescope ZIP, excluding all three encoder files')
    mode.add_argument('--encoder-only', action='store_true', help='Separate ZIP of three encoder files and LICENSE.md')
    args = parser.parse_args()
    package(args.out, args.encoder_dir, no_encoder=args.no_encoder, encoder_only=args.encoder_only)


if __name__ == '__main__':
    main()
