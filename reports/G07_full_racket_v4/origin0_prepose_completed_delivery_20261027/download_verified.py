"""Fresh-download the two uploaded artifacts without exposing GitHub credentials or signed URLs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VIDEO_REPORT = ROOT.parent / 'unitree_smash_video_framing/reports/G07_full_racket_v4/origin0_cold_four_contacts_video_20261027'
TAG = 'g07-full-racket-v4-progress-20261024'


def main():
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    spec = importlib.util.spec_from_file_location('prior_verified_transport',
        HERE.parent / 'contact_aim7_completed_delivery_20261027/download_verified.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    expected = [json.loads((VIDEO_REPORT / 'DECODE_AND_NATIVE_INPUT_CHECK.json').read_text()),
                json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())]
    receipts = []
    for number, check in enumerate(expected):
        name = Path(check.get('video') or check['archive']).name
        result = subprocess.run(['gh', 'api', 'repos/lzqw/unitree-smash/releases/422477709'],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
        assets = [a for a in json.loads(result.stdout)['assets'] if a['name'] == name]
        assert len(assets) == 1
        asset = assets[0]
        assert asset['state'] == 'uploaded' and asset['size'] == check['bytes']
        assert asset['digest'] == 'sha256:' + check['sha256']
        transport.ASSET_ID = asset['id']
        location = transport.signed_url()
        size = check['bytes']
        directory = Path(tempfile.mkdtemp(prefix='g07_origin0_completed_fresh_'))
        partial = directory / (name + '.partial')
        descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.ftruncate(descriptor, size)
        chunk = 32 * 1024 * 1024

        def part(start):
            end = min(size - 1, start + chunk - 1)
            for attempt in range(4):
                try:
                    request = urllib.request.Request(location, headers={
                        'Range': f'bytes={start}-{end}', 'User-Agent': 'unitree-smash-artifact-verification'})
                    with urllib.request.urlopen(request, timeout=25) as response:
                        assert response.status == 206
                        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                        offset = start
                        while offset <= end:
                            data = response.read(min(1024 * 1024, end - offset + 1))
                            assert data and offset + len(data) <= end + 1
                            assert os.pwrite(descriptor, data, offset) == len(data)
                            offset += len(data)
                        assert not response.read(1)
                    return end - start + 1
                except Exception as error:
                    if attempt == 3:
                        raise RuntimeError(f'Artifact range failed: {type(error).__name__}') from None
                    time.sleep(attempt + 1)

        downloaded = 0
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(part, start) for start in range(0, size, chunk)]):
                    downloaded += future.result()
                    print(json.dumps(dict(asset=name, downloaded_bytes=downloaded, total_bytes=size)), flush=True)
        finally:
            os.close(descriptor)
        assert downloaded == size and partial.stat().st_size == size
        assert hashlib.sha256(partial.read_bytes()).hexdigest() == check['sha256']
        payload_count = 0
        if number == 1:
            with zipfile.ZipFile(partial) as zipped:
                assert zipped.testzip() is None
                manifest = json.loads(zipped.read('MANIFEST.json'))
                payloads = manifest['payloads']
                assert len(zipped.namelist()) == len(set(zipped.namelist())) == len(payloads) + 1
                assert set(zipped.namelist()) == set(payloads) | {'MANIFEST.json'}
                for member, record in payloads.items():
                    content = zipped.read(member)
                    assert len(content) == record['bytes']
                    assert hashlib.sha256(content).hexdigest() == record['sha256'], member
                assert manifest['original128_cold_summary']['complete'] == 0
                assert manifest['original128_cold_summary']['airborne_contacts'] == 4
                assert all(f'search8/episodes_{i:03d}.json' in payloads for i in range(8))
                assert all(f'search8/all_first_episodes_{i:03d}.npz' in payloads for i in range(8))
                assert manifest['current_video_file'] in payloads and not manifest['formal_acceptance']
                payload_count = len(payloads)
        final = directory / name
        partial.rename(final)
        metadata = {k: asset[k] for k in ['id', 'name', 'browser_download_url', 'size', 'state', 'digest']}
        (HERE / ('GITHUB_VIDEO_ASSET.json' if number == 0 else 'GITHUB_ARCHIVE_ASSET.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        receipts.append(dict(asset=name, id=asset['id'], url=asset['browser_download_url'],
            bytes=size, sha256=check['sha256'], fresh_download=str(final), fresh_download_verified=True,
            archive_members_CRC_verified=payload_count + 1 if payload_count else None,
            archive_payload_SHA_verified=payload_count if payload_count else None))
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        tag=TAG, assets=receipts, upload_succeeded=True, fresh_downloads_verified=True,
        actual1024_search_plus128_cold_delivered=True, all_original_cases_retained=True,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(download_failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
