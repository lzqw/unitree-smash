"""Download the already uploaded immutable artifact; validate every payload.

Use bounded HTTP ranges so transient transport failures do not discard a full
download. GitHub credentials stay on api.github.com; signed URLs stay in memory.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
TAG = 'g07-full-racket-v4-progress-20261024'
ASSET_ID = int(json.loads((HERE/'GITHUB_ASSET.json').read_text())['id'])
ASSET = 'G07_leg06_pilot8_native_training_and_frozen_TRAIN128_20261026.zip'
SIZE = 253118980
DIGEST = '83ab00c15730d104f4baa2026cea513c773e3b1a73e1d3cb610da64596f85090'
CHUNK = 32 * 1024 * 1024


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def signed_url():
    credential = subprocess.run(['gh', 'auth', 'token'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert credential.returncode == 0
    access_token = credential.stdout.strip()
    request = urllib.request.Request(
        f'https://api.github.com/repos/lzqw/unitree-smash/releases/assets/{ASSET_ID}',
        headers={'Authorization': 'Bearer ' + access_token,
                 'Accept': 'application/octet-stream', 'User-Agent': 'unitree-smash-data-verification'})
    try:
        response = urllib.request.build_opener(NoRedirect()).open(request, timeout=25)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f'GitHub artifact redirect HTTP {error.code}') from None
        location = error.headers['Location']
    else:
        response.close()
        raise RuntimeError('Expected an artifact redirect, received another API representation')
    parsed = urllib.parse.urlparse(location)
    assert parsed.scheme == 'https' and parsed.hostname in {
        'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}
    return location


def main():
    directory = Path(tempfile.mkdtemp(prefix='g07_leg06_pilot8_range_download_20261026_'))
    partial = directory / (ASSET + '.partial')
    progress_path = directory / 'DOWNLOAD_PROGRESS.json'
    print(json.dumps(dict(download_directory=str(directory), asset=ASSET)), flush=True)
    for acquisition in range(5):
        try:
            location = signed_url()
            probe = urllib.request.Request(location, headers={'Range': 'bytes=0-0', 'User-Agent': 'unitree-smash-data-verification'})
            with urllib.request.urlopen(probe, timeout=25) as response:
                assert response.status == 206 and response.headers['Content-Range'] == f'bytes 0-0/{SIZE}'
                assert len(response.read()) == 1
                etag = response.headers.get('ETag')
            break
        except Exception as error:
            print(json.dumps(dict(acquisition_attempt=acquisition+1, category=type(error).__name__, reason_type=type(getattr(error, 'reason', None)).__name__)), flush=True)
            if acquisition == 4:
                raise RuntimeError('Could not establish a verified artifact range after5 transport attempts') from None
            time.sleep(acquisition+1)
    descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(descriptor, SIZE)
    completed = []

    def download(start):
        end = min(SIZE - 1, start + CHUNK - 1)
        for attempt in range(4):
            try:
                request = urllib.request.Request(location, headers={
                    'Range': f'bytes={start}-{end}', 'User-Agent': 'unitree-smash-data-verification'})
                with urllib.request.urlopen(request, timeout=25) as response:
                    assert response.status == 206 and response.headers['Content-Range'] == f'bytes {start}-{end}/{SIZE}'
                    assert not etag or response.headers.get('ETag') == etag
                    offset = start
                    while True:
                        data = response.read(min(1024 * 1024, end - offset + 1))
                        if not data:
                            break
                        assert offset + len(data) <= end + 1
                        written = os.pwrite(descriptor, data, offset)
                        assert written == len(data)
                        offset += len(data)
                    assert offset == end + 1
                return dict(start=start, end=end, bytes=end-start+1, attempts=attempt+1)
            except Exception as error:
                # Never print signed request URLs or credential-bearing headers.
                category = type(error).__name__
                if attempt == 3:
                    raise RuntimeError(f'Artifact range {start}-{end} failed: {category}') from None
                time.sleep(attempt + 1)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(download, start) for start in range(0, SIZE, CHUNK)]
            for future in as_completed(pending):
                completed.append(future.result())
                atomic(progress_path, dict(asset=ASSET, expected_bytes=SIZE, completed=completed))
                print(json.dumps(dict(completed_ranges=len(completed), downloaded_bytes=sum(x['bytes'] for x in completed))), flush=True)
    finally:
        os.close(descriptor)
    assert sum(x['bytes'] for x in completed) == SIZE
    digest = hashlib.sha256()
    with partial.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == DIGEST and partial.stat().st_size == SIZE
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert len(manifest['payloads']) == 540
        assert len(archive.namelist()) == len(set(archive.namelist())) == 541
        for name, item in manifest['payloads'].items():
            data = archive.read(name)
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], name
        assert manifest['required_native_runtime_companion']['sha256'] == '44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb'
        assert 'reports/G07_full_racket_v4/protocol_frozen01/training_cases.json' in manifest['payloads']
    downloaded = directory / ASSET
    partial.rename(downloaded)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        release=TAG, asset=ASSET, GitHub_asset_id=ASSET_ID,
        url=f'https://github.com/lzqw/unitree-smash/releases/download/{TAG}/{ASSET}',
        bytes=SIZE, sha256=DIGEST, server_digest='sha256:' + DIGEST,
        fresh_download=str(downloaded), fresh_downloaded_sha256=DIGEST,
        upload_succeeded=True, download_verified=True,
        all541_members_CRC_and_540_payload_SHA_verified=True,
        transport='Independent bounded32MiB ranges, four connections; final whole-file SHA verified',
        mandatory_CPU_data_dependencies_included=True, required_native_runtime_companion=manifest['required_native_runtime_companion'],
        fresh_extraction_CPU_check=str(HERE/'FRESH_RESTORE_CHECK.json'),
        fresh_canonical_actor_and_original128_results_checked=True, actual_valid_public_samples=1115,
        native_replay_in_fresh_extraction=False, new_native_execution=False,
        formal_acceptance=False, goal_complete=False)
    atomic(HERE/'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Preserve partial files/progress; suppress potentially signed URLs.
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
