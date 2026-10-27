"""Fresh-download completed swing9/cold128 ZIP and verify every payload."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]


def main():
    expected = json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())
    asset = json.loads((HERE / 'GITHUB_ARCHIVE_ASSET.json').read_text())
    assert asset['name'] == Path(expected['archive']).name and asset['size'] == expected['bytes']
    assert asset['state'] == 'uploaded' and asset['digest'] == 'sha256:' + expected['sha256']
    source = MAIN / 'reports/G07_full_racket_v4/contact_aim7_completed_delivery_20261027/download_verified.py'
    spec = importlib.util.spec_from_file_location('verified_github_transport', source)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ASSET_ID = asset['id']
    for attempt in range(5):
        try:
            location = transport.signed_url()
            request = urllib.request.Request(location, headers={'Range': 'bytes=0-0', 'User-Agent': 'unitree-smash-verification'})
            with urllib.request.urlopen(request, timeout=25) as response:
                assert response.status == 206 and response.headers['Content-Range'] == f"bytes 0-0/{expected['bytes']}"
                assert response.read() == b'P'
            break
        except Exception as error:
            print(json.dumps(dict(acquisition_attempt=attempt + 1, category=type(error).__name__)), flush=True)
            if attempt == 4:
                raise RuntimeError('Could not acquire release artifact after five attempts') from None
            time.sleep(attempt + 1)
    directory = Path(tempfile.mkdtemp(prefix='g07_swing9_fresh_download_'))
    chunk_size = 32 * 1024 * 1024

    def download(start):
        end = min(expected['bytes'] - 1, start + chunk_size - 1)
        for attempt in range(4):
            try:
                request = urllib.request.Request(location, headers={
                    'Range': f'bytes={start}-{end}', 'User-Agent': 'unitree-smash-verification'})
                with urllib.request.urlopen(request, timeout=25) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f"bytes {start}-{end}/{expected['bytes']}"
                    data = response.read(end - start + 2)
                assert len(data) == end - start + 1
                path = directory / f'chunk_{start:012d}'
                path.write_bytes(data)
                return start, path
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError('Bounded artifact range failed: ' + type(error).__name__) from None
                time.sleep(attempt + 1)
    chunks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(download, start) for start in range(0, expected['bytes'], chunk_size)]
        for future in as_completed(pending):
            chunks.append(future.result())
            print(json.dumps(dict(downloaded_ranges=len(chunks))), flush=True)
    target = directory / asset['name']
    with target.open('xb') as stream:
        for _, path in sorted(chunks):
            with path.open('rb') as chunk:
                shutil.copyfileobj(chunk, stream)
    assert target.stat().st_size == expected['bytes']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected['sha256']
    extracted = directory / 'extracted'
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        payloads = manifest['payloads']
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(payloads) + 1 == expected['members']
        assert set(archive.namelist()) == set(payloads) | {'MANIFEST.json'}
        for name, record in payloads.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data = archive.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
        archive.extractall(extracted)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        release='g07-full-racket-v4-progress-20261024', asset=asset['name'], GitHub_asset_id=asset['id'],
        url=asset['browser_download_url'], bytes=expected['bytes'], sha256=expected['sha256'],
        fresh_download=str(target), fresh_extracted=str(extracted), upload_succeeded=True, download_verified=True,
        fresh_downloaded_sha256=expected['sha256'], members_CRC_verified=len(payloads) + 1,
        payload_SHA_verified=len(payloads), controller_search_not_new_actor_training=manifest['controller_search_not_new_actor_training'],
        original128_cold_summary=manifest['original128_independent_cold_summary'],
        required_runtime_companion=manifest['required_runtime_companion'],
        source_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_reexecution=False, formal_acceptance=False, goal_complete=False)
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['bytes', 'sha256', 'members_CRC_verified', 'payload_SHA_verified', 'fresh_extracted']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
