"""Fresh-download the uploaded historical success video and verify exact bytes."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = 'lzqw/unitree-smash'
TAG = 'g07-full-racket-v4-progress-20261024'
ASSET_ID = 680219372


def main():
    expected = json.loads((HERE / 'FINAL_VIDEO.json').read_text())
    checked = json.loads((HERE / 'DECODE_AND_SCALE_CHECK.json').read_text())
    assert checked['selected_TRAIN29_complete_success']
    assert checked['source_is_historical_origin16']
    assert not checked['current_origin0_success_claimed']
    metadata_path = HERE / 'GITHUB_VIDEO_ASSET.json'
    metadata = json.loads(metadata_path.read_text())
    name = Path(expected['video']).name
    assert metadata['id'] == ASSET_ID and metadata['name'] == name
    assert metadata['size'] == expected['bytes'] and metadata['state'] == 'uploaded'
    assert metadata['digest'] == 'sha256:' + expected['sha256']
    source = Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v4/contact_aim7_completed_delivery_20261027/download_verified.py')
    spec = importlib.util.spec_from_file_location('verified_github_transport', source)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ASSET_ID = ASSET_ID
    target = Path(tempfile.mkdtemp(prefix='g07_normal_ratio_success_download_')) / name
    for attempt in range(5):
        try:
            location = transport.signed_url()
            request = urllib.request.Request(location, headers={'User-Agent': 'unitree-smash-video-verification'})
            with urllib.request.urlopen(request, timeout=25) as response:
                assert response.status == 200
                data = response.read(expected['bytes'] + 1)
            assert len(data) == expected['bytes']
            assert hashlib.sha256(data).hexdigest() == expected['sha256']
            target.write_bytes(data)
            break
        except Exception as error:
            print(json.dumps({'attempt': attempt + 1, 'error_category': type(error).__name__}), flush=True)
            if attempt == 4:
                raise RuntimeError('Video fresh download failed after five bounded attempts') from None
            time.sleep(attempt + 1)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        repo=REPO, release=TAG, asset=name, GitHub_asset_id=ASSET_ID,
        url=metadata['browser_download_url'], bytes=len(data), sha256=expected['sha256'],
        fresh_download=str(target), fresh_downloaded_sha256=hashlib.sha256(data).hexdigest(),
        upload_succeeded=True, download_verified=True,
        fresh_download_bytes_identical_to_all741_frame_checked_video=True,
        source_is_historical_origin16=True, current_origin0_success_claimed=False,
        new_native_execution=False, formal_acceptance=False,
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_transport_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert not (HERE / 'UPLOAD_RECEIPT.json').exists()
    transport.atomic(HERE / 'UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'failed': True, 'category': type(error).__name__}), flush=True)
        raise SystemExit(1) from None
