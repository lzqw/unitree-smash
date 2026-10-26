"""Fresh download through release asset API; never print tokens or signed URLs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import zipfile

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'UPLOAD_RECEIPT.json').exists()
    video = ROOT/'reports/G07_full_racket_v4/video/Current_effect_origin0_actual_failures_20261027.mp4'
    archive = HERE/'G07_origin0_two_completed_candidates_TRAIN128_video_evidence_20261027.zip'
    fresh = Path(tempfile.mkdtemp(prefix='g07_origin0_video_delivery_fresh_20261027_'))
    token = subprocess.check_output(['gh','auth','token'], text=True).strip()
    receipts = []
    for local in [video, archive]:
        query = '.[] | select(.name == '+json.dumps(local.name)+') | {id,name,size,digest,browser_download_url,state}'
        result = subprocess.run(['gh','api','repos/lzqw/unitree-smash/releases/422477709/assets?per_page=100',
                                 '--paginate','--jq',query], capture_output=True, text=True, timeout=90)
        assert result.returncode == 0
        metadata = json.loads(result.stdout)
        assert metadata['state'] == 'uploaded' and metadata['size'] == local.stat().st_size
        assert metadata['digest'] == 'sha256:'+sha(local)
        (HERE/('GITHUB_VIDEO_ASSET.json' if local.suffix == '.mp4' else 'GITHUB_ARCHIVE_ASSET.json')).write_text(json.dumps(metadata, indent=2)+'\n')
        url = f"https://api.github.com/repos/lzqw/unitree-smash/releases/assets/{metadata['id']}?fresh={time.time_ns()}"
        response = requests.get(url, headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'},
                                allow_redirects=False, stream=True, timeout=(20,60))
        if response.status_code in (301,302,303,307,308):
            location = response.headers['Location']; host = urlsplit(location).hostname or ''
            assert host.endswith('.githubusercontent.com') and urlsplit(location).scheme == 'https'
            response.close()
            response = requests.get(location, stream=True, timeout=(20,60))
        assert response.status_code == 200, 'Unexpected download HTTP status '+str(response.status_code)
        destination = fresh/local.name
        started = last = time.monotonic(); received = 0
        with destination.open('xb') as stream:
            for block in response.iter_content(chunk_size=1024*1024):
                if block:
                    stream.write(block); received += len(block)
                assert received <= metadata['size'] and time.monotonic()-started <= 900
                if time.monotonic()-last >= 25:
                    print(json.dumps(dict(name=local.name,received_bytes=received,total_bytes=metadata['size'])), flush=True)
                    last = time.monotonic()
        response.close()
        assert received == metadata['size'] and sha(destination) == sha(local)
        receipt = dict(metadata, fresh_download=str(destination), sha256=sha(destination), fresh_download_verified=True)
        if local.suffix == '.zip':
            with zipfile.ZipFile(destination) as zipped:
                assert zipped.testzip() is None
                manifest = json.loads(zipped.read('MANIFEST.json'))
                assert len(zipped.namelist()) == len(manifest['payloads'])+1
                for name, record in manifest['payloads'].items():
                    content = zipped.read(name)
                    assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
                receipt.update(all_CRC_passed=True,members=len(zipped.namelist()),payloads_SHA_verified=len(manifest['payloads']))
        receipts.append(receipt)
        print(json.dumps(dict(name=local.name,bytes=received,sha256=sha(destination),verified=True)), flush=True)
    value = dict(checked_at=datetime.now(timezone.utc).isoformat(), assets=receipts,
        actual_video_frames=json.loads(video.with_suffix('.json').read_text())['frames'],actual_video_duration_s=json.loads(video.with_suffix('.json').read_text())['duration_s'],formal_cases_read=False,formal_acceptance=False)
    (HERE/'UPLOAD_RECEIPT.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(fresh_downloads_verified=2,receipt=str(HERE/'UPLOAD_RECEIPT.json'))), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # HTTP exceptions can include expiring credentials from a redirect.
        print('Fresh download verification failed: '+type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
