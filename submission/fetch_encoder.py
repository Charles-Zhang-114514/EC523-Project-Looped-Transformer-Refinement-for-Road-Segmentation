#!/usr/bin/env python3
"""Acquire and verify the pinned DINOv3 encoder before offline inference.

Local directory and file:// sources work without a network. HTTP(S) and the
explicit Hugging Face option are preparation steps, never inference steps.
Files are staged and hashed before being published; failed stages are removed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

REPO_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
EXPECTED_SHA256 = {
    'config.json': '135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5',
    'preprocessor_config.json': '960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7',
    'model.safetensors': 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179',
}
FILES = (*EXPECTED_SHA256, 'LICENSE.md')
CACHE_NAME = '.encoder_verified.json'
ENCODER_BYTES = 1_212_568_641  # Three pinned files plus the upstream license.
MAX_ZIP_BYTES = 2 * ENCODER_BYTES + 8 * 1024**2
DEFAULT_DEST = Path(__file__).resolve().parent / 'encoder'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _stamp(directory):
    result = {}
    for name in EXPECTED_SHA256:
        info = (directory/name).stat()
        # ctime additionally invalidates a replaced file with a preserved mtime.
        result[name] = [info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    return result


def verify_encoder(directory, *, use_cache=True, write_cache=True):
    """Return pinned hashes; prediction never downloads or repairs any files."""
    directory = Path(directory)
    hint = 'Run python submission/fetch_encoder.py --dest <ENCODER_DIR> with one source option first.'
    try:
        stamps = _stamp(directory)
        if any(not (directory/name).is_file() for name in EXPECTED_SHA256):
            raise ValueError('Encoder entries must be regular readable files')
    except (OSError, ValueError) as error:
        raise ValueError(f'Missing or unreadable encoder files. {hint}') from error
    expected_cache = dict(sha256=dict(EXPECTED_SHA256), files=stamps)
    if use_cache:
        try:
            if json.loads((directory/CACHE_NAME).read_text()) == expected_cache:
                return dict(EXPECTED_SHA256)
        except (OSError, ValueError):
            pass
    for name, expected in EXPECTED_SHA256.items():
        actual = digest(directory/name)
        if actual != expected:
            raise ValueError(f'Encoder SHA-256 mismatch for {name}: expected {expected}, got {actual}. {hint}')
    if _stamp(directory) != stamps:
        raise ValueError(f'Encoder changed during verification. {hint}')
    if write_cache:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.verify-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(expected_cache, stream, sort_keys=True)
                stream.write('\n')
            os.replace(temporary, directory/CACHE_NAME)
        except OSError:
            # Read-only encoder directories remain usable, with a full rehash.
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(EXPECTED_SHA256)


def download_zip(url, destination):
    """Stream a direct HTTP(S) or local file URL, reporting byte progress."""
    if urlparse(url).scheme not in ('http', 'https', 'file'):
        raise ValueError('--from-url requires an HTTP(S) or file:// URL')
    downloaded, last = 0, 0.
    with urlopen(url, timeout=60) as response, Path(destination).open('wb') as stream:
        total = int(response.headers.get('Content-Length') or 0)
        if total > MAX_ZIP_BYTES:
            raise ValueError('Encoder ZIP exceeds the preparation size limit')
        while chunk := response.read(4*1024**2):
            downloaded += len(chunk)
            if downloaded > MAX_ZIP_BYTES:
                raise ValueError('Encoder ZIP exceeds the preparation size limit')
            stream.write(chunk)
            now = time.monotonic()
            if now-last >= 1:
                print(f'DOWNLOAD {downloaded} bytes' + (f' / {total} ({downloaded/total:.1%})' if total else ''), flush=True)
                last = now
    if total and downloaded != total:
        raise ValueError('Incomplete encoder ZIP download')
    print(f'DOWNLOAD complete: {downloaded} bytes', flush=True)


def unpack_encoder(archive, destination):
    """Copy exactly four named regular members; never trust archive paths."""
    with zipfile.ZipFile(archive) as zipped:
        selected = {}
        for info in zipped.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe path or symlink in encoder ZIP')
            if info.is_dir():
                continue
            if path.name not in FILES or path.name in selected:
                raise ValueError('Encoder ZIP must contain exactly the four unique encoder files')
            selected[path.name] = info
        if set(selected) != set(FILES):
            raise ValueError('Encoder ZIP is missing a required encoder file or LICENSE.md')
        if sum(info.file_size for info in selected.values()) > 2*ENCODER_BYTES:
            raise ValueError('Expanded encoder ZIP exceeds the preparation size limit')
        for name in FILES:
            with zipped.open(selected[name]) as source, (destination/name).open('wb') as target:
                shutil.copyfileobj(source, target, length=4*1024**2)


def fetch_encoder(dest, *, from_dir=None, from_url=None, from_hf=False):
    """Stage, verify all three hashes, then replace destination files."""
    if sum((from_dir is not None, from_url is not None, bool(from_hf))) != 1:
        raise ValueError('Choose exactly one of --from-dir, --from-url or --from-hf')
    dest = Path(dest).absolute()
    try:
        verify_encoder(dest, use_cache=False)
        if (dest/'LICENSE.md').is_file():
            print('SKIP: existing encoder files match all pinned SHA-256 values.', flush=True)
            return False
    except ValueError:
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    required = ENCODER_BYTES * (3 if from_url is not None else 1) + 64*1024**2
    if shutil.disk_usage(dest.parent).free < required:
        raise OSError(f'Insufficient encoder preparation space: need {required} free bytes')
    with tempfile.TemporaryDirectory(prefix='.encoder-fetch-', dir=dest.parent) as directory:
        stage = Path(directory)/'files'
        stage.mkdir()
        if from_dir is not None:
            for name in FILES:
                shutil.copyfile((Path(from_dir)/name).resolve(strict=True), stage/name)
        elif from_url is not None:
            archive = Path(directory)/'encoder.zip'
            download_zip(from_url, archive)
            unpack_encoder(archive, stage)
        else:
            from huggingface_hub import hf_hub_download
            for name in FILES:
                hf_hub_download(repo_id=REPO_ID, filename=name, revision=REVISION, local_dir=str(stage))
        verify_encoder(stage, use_cache=False, write_cache=False)
        if not (stage/'LICENSE.md').is_file():
            raise ValueError('Encoder source must include LICENSE.md')
        dest.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            os.replace(stage/name, dest/name)
        verify_encoder(dest, use_cache=False)
    for name, value in EXPECTED_SHA256.items():
        print(f'VERIFIED {name} sha256={value}', flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-dir', type=Path, help='Copy dereferenced local encoder files')
    source.add_argument('--from-url', help='Direct encoder ZIP URL; file:// is supported offline')
    source.add_argument('--from-hf', action='store_true', help='Download the pinned gated HF revision using your existing login')
    args = parser.parse_args(argv)
    try:
        fetch_encoder(args.dest, from_dir=args.from_dir, from_url=args.from_url, from_hf=args.from_hf)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Encoder preparation failed: {error}\n')


if __name__ == '__main__':
    main()
